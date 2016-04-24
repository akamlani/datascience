import java.sql.Timestamp
import java.text.SimpleDateFormat
import scala.math.Ordering.Implicits._
import java.util.{Date, Calendar}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object movielens {
  val packagePath = "/Users/akamlani/Projects/datascience/packages/spark/data/movielens/"
  case class movieRatings(userid: Long, movieid: Long, rating: Double, timestamp: Timestamp)
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("movielens").setMaster("local[*]")
    val sc = new SparkContext(conf)

    //Date object is deprecated, used Calandar object (http://docs.oracle.com/javase/6/docs/api/java/util/Date.html)
    val cal = Calendar.getInstance()
    val sdf = new SimpleDateFormat("yyyy-MM-dd");

    def transformTimestamp(movieRatingsRDDIn: RDD[movieRatings], rddQuery: String):
    Either[RDD[(Date, Long)], RDD[((Int, Int), Long)]]  = {
      rddQuery match {
        //map by ((Date), userid)
        case "userDate" =>
          val tsDateUsersTransfRDD: RDD[(Date, Long)] = movieRatingsRDDIn.map { obj =>
            cal.setTime(obj.timestamp)
            (cal.getTime(), obj.userid)
          }
          Left(tsDateUsersTransfRDD)
        //map by ((year,month), userid)
        case "userYearMonth" =>
          val tsYearMonthUsersTransfRDD: RDD[((Int, Int), Long)] = movieRatingsRDDIn.map { obj =>
            cal.setTime(obj.timestamp)
            val  year   = cal.get(Calendar.YEAR);
            val  month  = cal.get(Calendar.MONTH) + 1;      //returns sequence {0-11}
            val  day    = cal.get(Calendar.DAY_OF_MONTH);   //returns sequence {0-6}
            ((year, month), obj.userid)
          }
          Right(tsYearMonthUsersTransfRDD)
        /*
        //map by (month, userid)
        case "userMonth" =>
          //map by (month, userid)
          val tsMonthUsersTransfRDD: RDD[(Int, Long)] = movieRatingsRDDIn.map { obj =>
            cal.setTime(obj.timestamp)
            val  year   = cal.get(Calendar.YEAR);
            val  month  = cal.get(Calendar.MONTH) + 1;      //returns sequence {0-11}
            val  day    = cal.get(Calendar.DAY_OF_MONTH);   //returns sequence {0-6}
            (month, obj.userid)
          }
        */
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
      val tsUsersRDD: RDD[((Int, Int), Long)] = transformTimestamp(movieRatingsRDDIn, "userYearMonth").right.toOption.get
      //Calculate two different metrics for new users: 1. by (Year,Month), 2. by Month
      //determine the (year,month) in sorted order of the number of times a new user appears
      //for each user extract only the earliest time sequence (year, month)
      //note the avoidance of grouping the users and extracting the the length to avoid entire RDD sequence in memory
      val newUsersTsRDD: RDD[((Int, Int), Int)] = tsUsersRDD.map{case (yearMonth, userid) =>
        (userid, yearMonth)
      }.reduceByKey( (a,b) => if(a < b) a else b)
        .map{ case (userid, yearMonth) =>
          (yearMonth, 1)
        }.reduceByKey((x,y) => x + y)
        .sortByKey(ascending=true)
        .cache()

      //next determine by month of new users
      val newUsersViaMonthRDD = newUsersTsRDD.map{ case ((year,month), newUsers) =>
        (month, newUsers)
      }.reduceByKey((a,b) => a + b)

      //Collect results and save output
      println("Number of new users per (year, month): ")
      newUsersTsRDD.collect.foreach(println)
      val header1: RDD[String] = sc.parallelize(Array("Year\tMonth\tUserCount"))
      header1.union {
        newUsersTsRDD
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

      //mark the RDD as no required to be kept in memory
      newUsersTsRDD.unpersist()
    }

    def kpi2CalculateNumInactiveUsers(movieRatingsRDDIn: RDD[movieRatings]) = {
      //KPI 2: determine number of users that are leaving (assuming being inactive for 3 months)
      //DECISION: Inactive = from Latest Date to maximum Date >= 3 Months
      //for each user time slice period, determine if the latest date has been a 3 month inactive period
      val maxDate: Date = movieRatingsRDDIn.map{ obj =>
        cal.setTime(obj.timestamp)
        sdf.parse( sdf.format(cal.getTime()) );
      }.max();

      val tsUsersRDD: RDD[(Date, Long)] = transformTimestamp(movieRatingsRDDIn, "userDate").left.toOption.get
      //get the latest date a user rated out of all of their ratings; excluding HH:MM:SS
      //note the usage of reduceByKey in comparison to Date objects rather than grouping by the key and sorting
      val mostRecentUserRatingRDD: RDD[(Long, Date)] = tsUsersRDD.map{case (date, userid) =>
        cal.setTime(date)
        (userid, sdf.parse(sdf.format(date)) )
      }.reduceByKey((a,b) => if(a.compareTo(b) > 0) a else b )
        .cache()

      //determine if the last known date is greater than 3 Months
      //if values is 1 => LEAVING, else staying (-1,0): Note that 0 is interpret as equivalent to same as Max Date
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
      val header: RDD[String] = sc.parallelize(Array("UniqueUsers\tInactiveUsers\t % InactiveUsers"))
      header.union { sc.parallelize(Array(totalUniqueUsers.toString + "\t\t" +
                     totalInactiveUsers.toString + "\t\t" + pctInactiveUsers.toString))
      }.coalesce(1, true)
        .saveAsTextFile(packagePath + "/export/KPI_2_INACTIVE_USERS")

      //mark the RDD as no longer required in memory
      mostRecentUserRatingRDD.unpersist()
    }

    def kpi3CalculateNumRatingsPerMonth(movieRatingsRDDIn: RDD[movieRatings]) = {
      //KPI 3: determine the number of ratings mapped to a given month
      //we make a shortcut here since we know that any reported observation is guaranteed to be a rating made
      //so we just need to count the ratings per month
      val ratingsPerMonthRDD: RDD[(Int, Int)] = movieRatingsRDDIn.map { obj =>
        cal.setTime(obj.timestamp)
        val  month  = cal.get(Calendar.MONTH) + 1;      //returns sequence {0-11}
        (month, 1)
      }.reduceByKey((a,b) => a + b)
      println("Ratings Mapped to a given Month: ")
      ratingsPerMonthRDD.collect.foreach(println)
      val header: RDD[String] = sc.parallelize(Array("Month\tRatingCount"))
      header.union {
        ratingsPerMonthRDD.map { case (month, ratingsCnt) =>
          month.toString + "\t" + ratingsCnt.toString
        }
      }.coalesce(1, true)
        .saveAsTextFile(packagePath + "/export/KPI_3_RATINGS_BY_MONTH")
    }

    def kpi4CalculateStatsTopActiveUsers(movieRatingsRDDIn: RDD[movieRatings]) = {
      //KPI 4: Mean + Standard Deviation of 10% most active users based on the average of their ratings

      /* Step 1. Determine Most Active Users based on average of ratings:
          Implement with a narrow map rather than with a groupBy to avoid an additional wide operation calculation
          The wide operation takes much longer as it needs to place all in memory
          Hence we perform an addition of input keys rather than summing a list of a grouped key and getting the count
          Alternatively this can have been replaced by CombineByKey
       */
      val header: RDD[String] = sc.parallelize(Array("Mean\t\tStandard Deviation"))
      val topUserAvgRatingsRDD: RDD[((Long, Double), Long)] = movieRatingsRDDIn.map { obj =>
        (obj.userid, (obj.rating, 1))
      }.reduceByKey{(x, y) =>
        (x._1 + y._1, x._2 + y._2)
      }.mapValues { case (summation, counter) =>
        summation/counter
      }.sortBy(_._2, false).zipWithIndex()

      //Step 2. Cut down to include top 10% based on count and indexes
      //10% most active users (sorted)
      val top10Pct: Long = (topUserAvgRatingsRDD.count() * 0.10).round
      val top10PctRDD: RDD[((Long, Double), Long)] = topUserAvgRatingsRDD.filter{ case (v, idx) => idx < top10Pct}
      //Step 3. Calculate Mean + STD
      val top10PctMean: Float = top10PctRDD.map {case ((userid,avgRating), idx) => avgRating}.mean().toFloat
      val top10PctSTD: Float = top10PctRDD.map {case ((userid, avgRating), idx) => avgRating}.stdev().toFloat
      println( s"""Based on 10% Most Active Users: Mean=$top10PctMean, STD=$top10PctSTD""")
      //write to file for easy viewing
      header.union { sc.parallelize(Array(top10PctMean.toString + "\t" + top10PctSTD.toString)) }
        .coalesce(1, true)
        .saveAsTextFile(packagePath + "/export/KPI_4_ACTIVEUSERS_AVGMETRICS")

      /*
      ALT Step 1. Instead of calculate average of their ratings, implement based on frequency (how many times they rated)
      An even further alternative would be to determine based on a particular time scale (e.g. rated within the month)
      */
      val topUserFreqRatingsRDD = movieRatingsRDDIn.map { obj =>
        (obj.userid, 1)
      }.reduceByKey{(x, y) =>
        x + y
      }.sortBy(k => k._2, false).zipWithIndex()
      //Step 2. Cut down to include top 10% based on count frequency od ratings per user
      //10% most active users (sorted)
      val freqTop10Pct: Long = (topUserFreqRatingsRDD.count() * 0.10).round
      val freqTop10PctRDD: RDD[((Long, Int), Long)] = topUserFreqRatingsRDD.filter { case (v, idx) => idx < freqTop10Pct}
      //Step 3. Calculate Mean + STD
      val movieUserRatingsRDD: RDD[(Long, Double)] = movieRatingsRDDIn.map { obj =>  (obj.userid, obj.rating)}
      val metricsRDD: RDD[Double] = freqTop10PctRDD.map{case ((userid, counter),idx) =>
        (userid, counter)
      }.join(movieUserRatingsRDD)
        .map{case (userid, (cnt,rating)) => rating}

      val freqTop10PctMean: Float = metricsRDD.mean().toFloat
      val freqTop10PctSTD: Float  = metricsRDD.stdev().toFloat
      println( s"""Based on 10% Most Frequent Active Users: Mean=$freqTop10PctMean, STD=$freqTop10PctSTD""")
      //write to file for easy viewing
      header.union { sc.parallelize(Array(freqTop10PctMean.toString + "\t" + freqTop10PctSTD.toString)) }
        .coalesce(1, true)
        .saveAsTextFile(packagePath + "/export/KPI_4_ACTIVEUSERS_FREQMETRICS")
    }

    //Movielens 20m dataset is given for the years 1995-2015
    //Should correlate with: Max Date = (2015,3, 31), Min Date = (1995,1, 9)
    val movieRatingsRDD: RDD[movieRatings] = mapInputToUserRatings().cache()
    //perform KPI(i) calculations

    //Note that for KPI1, we implement two different metrics: new users via (Year,Month) and by Month
    kpi1CalculateNumNewUsersMetric(movieRatingsRDD)

    //Note the definition of INACTIVE in KPI2: (assumes > 3 Months for Max Date of Movielens 20M dataset)
    //The majority have been in INACTIVE for more than 3 Months!!!
    kpi2CalculateNumInactiveUsers(movieRatingsRDD)

    kpi3CalculateNumRatingsPerMonth(movieRatingsRDD)
    //Note that for KPI 4: we implement two different methods (based on user average ratings and freq of ratings)
    kpi4CalculateStatsTopActiveUsers(movieRatingsRDD)

    //!!!!!!Reference output data in /<root path>/data/movielens/export/KPI_X_*

    //mark the RDD as no longer required to be kept in memory
    movieRatingsRDD.unpersist()

  }
}