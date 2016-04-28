import org.apache.spark.{SparkConf, SparkContext}

object bank {
  case class Bank(age: Integer, job: String, marital: String, education: String, balance: Integer, loan: Boolean)
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("bank").setMaster("local[*]")
    val sc = new SparkContext(conf)

    // load bank data
    val packagePath = "/Users/akamlani/Projects/datascience/packages/spark/data/"
    val bankText = sc.textFile(packagePath + "bank.csv")
    //get information about the data
    val numobs = bankText.count
    val headers = bankText.first.split(";")
    //put together some indices of each relevant attribute to be used for the bank object
    val lineText = bankText.map(s => s.split(";"))
    val q = List("\"age\"", "\"job\"", "\"marital\"", "\"education\"", "\"balance\"", "\"loan\"")
    val pairs = q.map(k => (k,headers.indexOf(k)))


    /**** Part I: maninpulate/update bank object ****/
    def convertBoolean(line: String):Boolean = line match{
      case "yes" => true
      case "no" =>  false
    }
    //create a Bank Object and update based on loan status
    val file = lineText.filter(s => !s.contains("\"age\"")).map(
      s => Bank(s(0).toInt,
        s(1).replaceAll("\"", ""),
        s(2).replaceAll("\"", ""),
        s(3).replaceAll("\"", ""),
        s(5).replaceAll("\"", "").toInt,
        convertBoolean( s(7).replaceAll("\"", "").toLowerCase)
      ))
    
    /**** Part II: calculate % married that have a loan ****/
    def extractMarital(b:Bank) : (String) = {
      (b.marital)
    }
    def extractMaritalLoan(b:Bank) : (String, Boolean) = {
      (b.marital, b.loan)
    }
    //get the unique types of married status
    val maritalStatus = file.map(extractMarital(_)).distinct().collect()
    //group by marital status, filter on 'married' and extract loan count conditions, count the percentage for Loan is true
    val maritalStatusLoanStatus = file.map(extractMaritalLoan(_)).groupByKey()
    val marriedLoanStatus       = maritalStatusLoanStatus.filter(_._1 == "married").flatMap(_._2)
    val marriedLoanStatusMap    = marriedLoanStatus.countByValue
    val marriedLoanStatusPct    = (marriedLoanStatusMap.get(true).get.toDouble/marriedLoanStatus.count) * 100
    printf("\nPct Married that have a loan = %f", marriedLoanStatusPct)

    /**** Part III: the age that has max number of jobs ****/
    def extractAgeJob(b:Bank) : (Int, String) = {
      (b.age, b.job)
    }
    def extractJob(b: Bank) : (String) = {
      (b.job)
    }
    def extractAge(b: Bank) : (Int) = {
      (b.age)
    }

    //Calculate positions considered to have "jobs"
    //there are some statuses that we consider as not having qualified as "jobs"
    val jobStatus   = file.map(extractJob(_)).distinct().collect()
    val notJobs     = List("self-employed", "unknown", "unemployed", "retired", "student")
    val jobs        = jobStatus diff notJobs
    val withJobs    = file.map(extractAgeJob(_)).filter(x => jobs.find(_.contains(x._2)) != None )
    //Sort the jobs in descending order with most number of jobs
    val numWithJobs = withJobs.count
    val jobsByAge   = withJobs.groupByKey().map(x => (x._1, x._2.size))
    val ageMaxJobs  = jobsByAge.sortBy(-_._2).first
    printf("\nAge with Max Jobs-> Age: %d, NumJobs: %d\n", ageMaxJobs._1, ageMaxJobs._2)


    /**** Part IV ****/
    def extractMaritalBalance(b:Bank) : (String, Integer) = {
      (b.marital, b.balance)
    }

    //calculate the total average balance from (k,v pair)
    val avgBalance = file.map(extractMaritalBalance(_)._2).reduce(_ + _)/file.count
    //Try accumulator method to calculate total average Balance:
    val acc = sc.accumulator(0)
    file.map(extractMaritalBalance(_)._2 ).foreach(acc += _)
    val avgBalanceViaAcc = acc.value/file.count

    //CASE A: calculate if GROUP average balance is above or below OVERALL balance average
    val maritalStatusGroup = file.map(extractMaritalBalance(_)).groupByKey()
    val maritalStatusGroupAvgBalance = maritalStatusGroup.map(x => (x._1, (x._2).reduce(_+_)/(x._2).size))
    //Case A1. For each Marital.status group, which groups are above the Total Average
    val maritalStatusGroupElevatedOverallAvg = maritalStatusGroupAvgBalance.filter(x => x._2 > avgBalance).collect
    //Case A2. For each Marital.status group, which groups are below the Total Average
    val maritalStatusGroupBelowOverallAvg    = maritalStatusGroupAvgBalance.filter(x => x._2 < avgBalance).collect

    //log the info for each marital status as a group
    println("\nAverage Balance per Marital Status:")
    maritalStatusGroupAvgBalance.collect.foreach(x =>
      printf("marital status = %s, \t%s Avg Balance = %d, \tTotal Avg Balance=%d\n", x._1,x._1,x._2, avgBalance ) )


    //CASE B: calculate if balance is above or below GROUP average
    //Case B1: determine if the percentage of each group is above the group average
    println("\nPercentage Balance above or below GROUP average:")
    val keyMapGroupBalance = maritalStatusGroupAvgBalance.collect.toMap
    val maritalStatusElevated = maritalStatusGroup.map(x => (x._1,  (x._2).map(_ > keyMapGroupBalance.get(x._1).get) ) )
    val maritalStatusElevatedPct  = maritalStatusElevated.map(x =>
      (x._1, (x._2.filter(_ == true).size.toDouble/x._2.size.toDouble)*100) )
    //Case B2: determine if the percentage of each group is below the group average
    val maritalStatusBelowPct  = maritalStatusElevated.map(x =>
      (x._1, (x._2.filter(_ == false).size.toDouble/x._2.size.toDouble)*100) )


    //CASE C: calculate if balance is above or below OVERALL average
    //Case C1: determine if the balance is above overall average
    println("\nPercentage Balance above or below OVERALL average:")
    val maritalStatusElevatedOverall = maritalStatusGroup.map(x => (x._1,  (x._2).map(_ > avgBalance) ) )
    val maritalStatusElevatedOverallPct  = maritalStatusElevatedOverall.map(x =>
      (x._1, (x._2.filter(_ == true).size.toDouble/x._2.size.toDouble)*100) )
    //Case C2: determine if the balance is below overall average
    val maritalStatusBelowOverallPct  = maritalStatusElevatedOverall.map(x =>
      (x._1, (x._2.filter(_ == false).size.toDouble/x._2.size.toDouble)*100) )


  }
}