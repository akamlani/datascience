import java.sql.Timestamp
import java.text.SimpleDateFormat
import java.util.{GregorianCalendar, Date, Calendar}
import org.apache.spark.rdd.RDD
import org.apache.spark.{rdd, Accumulator, SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, SQLContext}

object movielens {
  val packagePath = "/Users/akamlani/Projects/datascience/packages/spark/data/movielens/"
  case class movieRatings(userid: Long, movieid: Long, rating: Double, timestamp: Timestamp)
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("movielens").setMaster("local[*]")
    val sc = new SparkContext(conf)

    //Date object is deprecated, used Calandar object (http://docs.oracle.com/javase/6/docs/api/java/util/Date.html)
    val cal = Calendar.getInstance()
    val sdf = new SimpleDateFormat("yyyy-MM-dd");


    def queryTransformTimestampRDD(movieRatingsRDDIn: RDD[movieRatings], rddQuery: String):
    Either[RDD[(Date, Long)], RDD[((Int, Int), Long)]]  = {
      //extract the month from the timestamp object and relate it to the user.id
      //map by ((Date), userid)
      val timeSliceFullUsersRDD: RDD[(Date, Long)] = movieRatingsRDDIn.map { obj =>
        cal.setTime(obj.timestamp)
        (cal.getTime(), obj.userid)
      }
      //map by ((year,month), userid)
      val timeSliceUsersRDD: RDD[((Int, Int), Long)] = movieRatingsRDDIn.map { obj =>
        cal.setTime(obj.timestamp)
        val  year   = cal.get(Calendar.YEAR);
        val  month  = cal.get(Calendar.MONTH) + 1;      //returns sequence {0-11}
        val  day    = cal.get(Calendar.DAY_OF_MONTH);   //returns sequence {0-6}
        ((year, month), obj.userid)
      }
      //map by (month, userid)
      val monthlySliceUsersRDD: RDD[(Int, Long)] = movieRatingsRDDIn.map { obj =>
        cal.setTime(obj.timestamp)
        val  year   = cal.get(Calendar.YEAR);
        val  month  = cal.get(Calendar.MONTH) + 1;      //returns sequence {0-11}
        val  day    = cal.get(Calendar.DAY_OF_MONTH);   //returns sequence {0-6}
        (month, obj.userid)
      }

      rddQuery match {
        case "userDateRDDMapped" =>  Left(timeSliceFullUsersRDD)
        case "userYearMonthRDDMapped" =>  Right(timeSliceUsersRDD)
        //case "userMonthlyRDDMapped" =>  Left(monthlySliceUsersRDD)
      }
    }


    def mapInputToUserRatings(): RDD[movieRatings] = {
      val ratingsBaseRDD  = sc.textFile(packagePath + "ml-20m/ratings.csv")
      //choose to operate as hardcoded known header rather than performing an action to retrieve the header
      //this will save an 'action', and continue with lazy evaluation
      val ratingsRDD = ratingsBaseRDD.filter{ line =>
        !line.contains("userId,movieId,rating,timestamp")
      }.map{ line =>
        val data = line.split(",")
        (data(0), data(1), data(2), data(3))
      }
      //map into a movieRatings class
      //Timestamps represent seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970.
      //Timestamps come back in YYYY-mm-day HH:MM:SS format
      //Note that timestamp needs 'ms' format - hence multiply by 1000
      val movieRatingsRDD: RDD[movieRatings] = ratingsRDD.map{ case(userid, movieid, rating, timestamp) =>
        movieRatings(userid = userid.toLong, movieid = movieid.toLong, rating = rating.toDouble,
          timestamp = new Timestamp(timestamp.toLong * 1000))
      }
      return movieRatingsRDD
    }


    def kpi1CalculateNumNewUsersMetric(movieRatingsRDDIn: RDD[movieRatings]) = {
      //KPI 1: determine number of new users every month

      //group by UserID and unique dates (year, month) in sorted order; no duplicate userids will be registered in the RDD
      //this gives the number of users per a date format (year, month) from their earliest occurence date = REGISTRATION
      val timeSliceUsersRDD: RDD[((Int, Int), Long)] =
        queryTransformTimestampRDD(movieRatingsRDDIn, "userYearMonthRDDMapped").right.toOption.get
      val usersTimeSliceRDD: RDD[(Long, List[(Int, Int)])] =
        timeSliceUsersRDD.map{case (times,userid) => (userid,times)}
          .groupByKey()
          .mapValues{times => times.toList.distinct.sorted}
          .cache()

      //Calculate two different metrics for new users: 1. by (Year,Month), 2. by Month
      //determine the (year,month) in sorted order of the number of times a new user appears
      //for each user extract only the earliest time sequence (year, month) => toList(0)
      val newUsersTimeSliceRDD: RDD[((Int, Int), Int)] = usersTimeSliceRDD.map{case (userid, times) =>
        (times.toList(0), userid)
      }.groupByKey().map{case (times, users) =>
        (times, users.toList.length)
      }.sortByKey(ascending=true)
      //determine by month of new users
      val newUsersViaMonthRDD = newUsersTimeSliceRDD.map{ case (times, newUsers) =>
        (times._2, newUsers)
      }.reduceByKey((a,b) => a + b)

      //Collect results and save output
      println("Number of new users per (year, month): ")
      newUsersTimeSliceRDD.collect.foreach(println)
      val header1: RDD[String] = sc.parallelize(Array("Year\tMonth\tUserCount"))
      header1.union {
        newUsersTimeSliceRDD
          .map { case ((year, month), userCnt) => year.toString + "\t" + month.toString + "\t" + userCnt.toString }
      }.coalesce(1, true)
        .saveAsTextFile(packagePath + "/export/KPI_1_BY_YEARMONTH")

      println("Number of new users every month: ")
      newUsersViaMonthRDD.collect.foreach(println);
      val header2: RDD[String] = sc.parallelize(Array("Month\tUserCount"))
      header2.union {
        newUsersViaMonthRDD.map { case (month, userCount) => month.toString + "\t" + userCount.toString }
      }.coalesce(1, true)
        .saveAsTextFile(packagePath + "/export/KPI_1_BY_MONTH")

      //no longer required to keep in memory for calculations
      usersTimeSliceRDD.unpersist()
    }

    def kpi2CalculateNumInactiveUsers(movieRatingsRDDIn: RDD[movieRatings]) = {
      //KPI 2: determine number of users that are leaving (assuming being inactive for 3 months)
      //DECISION: Inactive = from Latest Date to maximum Date >= 3 Months
      //for each user time slice period, determine if there has been a 3 month inactive period
      val maxDate: Date = movieRatingsRDDIn.map{ obj =>
        cal.setTime(obj.timestamp)
        sdf.parse( sdf.format(cal.getTime()) );
      }.max();

      val timeSliceFullUsersRDD =
        queryTransformTimestampRDD(movieRatingsRDDIn, "userDateRDDMapped").left.toOption.get

      //get the latest date a user rated out of all of their ratings
      val mostRecentUserRatingRDD: RDD[(Long, Date)] =
        timeSliceFullUsersRDD.map{case (times,userid) => (userid,times)}
          .groupByKey()
          .mapValues{times => times.toList.distinct.toList.sortWith(_ after _ )}
          .mapValues{times => times.toList(0)}
      //determine if the last known date is greater than 3 Months
      //if values is 1 => LEAVING, else staying (-1,0)
      val inactiveUsersRDD: RDD[(Long, Int)] = mostRecentUserRatingRDD.map {case (userid, times) =>
        //set the maximum date allowed for INACTIVITY
        cal.setTime(maxDate)
        cal.add(Calendar.MONTH, -3)
        val inactiveMaxDate: Date = cal.getTime()
        val userTime: Date = sdf.parse( sdf.format(times) );
        //If the Maximum Inactive Date > date arguement == 1 -> LEAVING
        (userid, inactiveMaxDate.compareTo(userTime))
      }.filter { case (userid, inactiveStatus) =>
        inactiveStatus > 0
      }

      //get the count of the inactive users
      val totalUniqueUsers: Long = mostRecentUserRatingRDD.count()
      val totalInactiveUsers: Long = inactiveUsersRDD.count()
      val pctInactiveUsers: Float = (totalInactiveUsers.toFloat/totalUniqueUsers) * 100
      println(s"""Total Users: $totalUniqueUsers""")
      println(s"""Inactive Users (Leaving): $totalInactiveUsers, %Leaving: $pctInactiveUsers""")
      val header: RDD[String] = sc.parallelize(Array("Unique Users\tInactive Users\t % Inactive Users"))
      header.union { sc.parallelize(Array(totalUniqueUsers.toString + "\t" +
                     totalInactiveUsers.toString + "\t" + pctInactiveUsers.toString))
      }.coalesce(1, true)
        .saveAsTextFile(packagePath + "/export/KPI_2_INACTIVE_USERS")
    }

    def kpi3CalculateNumRatingsPerMonth(movieRatingsRDDIn: RDD[movieRatings]) = {
      //KPI 3: determine the number of ratings mapped to a given month
      //we make a shortcut here since we know that any reported observation is guaranteed to be a rating made
      val ratingsPerMonthRDD: RDD[(Int, Int)] = movieRatingsRDDIn.map { obj =>
        cal.setTime(obj.timestamp)
        val  month  = cal.get(Calendar.MONTH) + 1;      //returns sequence {0-11}
        (month, 1)
      }.reduceByKey((a,b) => a + b)
      println("Ratings Mapped to a given Month: ")
      ratingsPerMonthRDD.collect.foreach(println)
      val header: RDD[String] = sc.parallelize(Array("Month\tUserCount"))
      header.union {
        ratingsPerMonthRDD.map { case (month, ratingsCnt) =>
          month.toString + "\t" + ratingsCnt.toString
        }
      }.coalesce(1, true)
        .saveAsTextFile(packagePath + "/export/KPI_3_RATINGS_BY_MONTH")
    }

    def kpi4CalculateStatsTopActiveUsers(movieRatingsRDDIn: RDD[movieRatings]) = {
      //KPI 4: Mean + Standard Deviation of 10% most active users based on the average of their ratings
      //Step 1. Determine Most Active Users based on average of ratings:
      val topUserRatingsRDD: RDD[((Long, Double), Long)] = movieRatingsRDDIn.map { obj => (obj.userid, obj.rating) }
        .groupByKey()
        .mapValues{ v =>
          val n = v.toList.length
          val sum = v.toList.sum
          (sum/n)
      }.sortBy(kv => kv._2, false).zipWithIndex()

      //Step 2. Cut down to include top 10% based on count
      //10% most active users (sorted vector count reduced to: 138493 -> 13849)
      val top10Pct: Long = (topUserRatingsRDD.count() * 0.10).round
      val top10PctRDD: RDD[((Long, Double), Long)] = topUserRatingsRDD.filter { case ((userid,rating), idx) => idx < top10Pct}
      //Step 3. Calculate Mean + STD
      val top10PctMean: Double = top10PctRDD.map {case ((userid,rating), idx) => rating}.mean()
      val top10PctSTD: Double = top10PctRDD.map {case ((userid,rating), idx) => rating}.stdev()
      println( s"""Based on 10% Most Active Users: Mean=$top10PctMean, STD=$top10PctSTD""")
      //write to file
      val header: RDD[String] = sc.parallelize(Array("Mean\tStandard Deviation"))
      header.union { sc.parallelize(Array(top10PctMean.toString + "\t" + top10PctSTD.toString)) }
        .coalesce(1, true)
        .saveAsTextFile(packagePath + "/export/KPI_4_ACTIVEUSERS_METRICS")
    }

    //Movielens 20m dataset is given for the years 1995-2015
    //Should correlate with: Max Date = (2015,3, 31), Min Date = (1995,1, 9)
    val movieRatingsRDD: RDD[movieRatings] = mapInputToUserRatings().cache()
    //perform KPI(i) calculations
    kpi1CalculateNumNewUsersMetric(movieRatingsRDD)
    kpi2CalculateNumInactiveUsers(movieRatingsRDD)
    kpi3CalculateNumRatingsPerMonth(movieRatingsRDD)
    kpi4CalculateStatsTopActiveUsers(movieRatingsRDD)

    movieRatingsRDD.unpersist()

  }
}