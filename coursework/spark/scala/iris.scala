//double value


import org.apache.spark.rdd.RDD
import org.apache.spark.util.StatCounter
import org.apache.spark.{Accumulator, SparkConf, SparkContext}

import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint


object iris {
  case class iris(sepalLength: Double, sepalWidth: Double, petalLength: Double, petalWidth: Double, species: String )

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("iris").setMaster("local[*]")
    val sc = new SparkContext(conf)
    println(sc.master)

    val packagePath = "/Users/akamlani/Projects/datascience/packages/spark/"

    //spark mllib implementation of naive bayes
    def naiveBayesSparkML(inputRDD: RDD[Array[String]]) = {
          val parsedData = inputRDD.map { parts =>
            //convert to labels case class
            val label_species = parts(4) match {
              case "Iris-versicolor" => 0
              case "Iris-virginica"  => 1
              case "Iris-setosa"     => 2
            }
            LabeledPoint( label_species.toDouble,
              Vectors.dense(parts(0).toDouble, parts(1).toDouble, parts(2).toDouble, parts(3).toDouble))
          }

          // Split data into training (60%) and test (40%).
          val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)
          val training = splits(0)
          val test = splits(1)
          val model = NaiveBayes.train(training, lambda = 1.0, modelType = "multinomial")

          val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))
          val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()
          println(s"Naive Bayes Model Accuracy = $accuracy")
    }

    //manual implmementation of statistics to calculate the spread of prediction
    //manual implementation of naive bayes to calculate std deviation and statistics
    def calculateStatsSpread(inputRDD: RDD[Array[String]]) = {
      val irisBaseRDD: RDD[iris] = inputRDD.map(data => iris(sepalLength = data(0).toDouble,
        sepalWidth = data(1).toDouble, petalLength = data(2).toDouble, petalWidth = data(3).toDouble, species = data(4)))
      //get number of observations
      println(s"number of observations = " + irisBaseRDD.count())

      //get the unique target labels
      val labels = irisBaseRDD.map(obj => obj.species).distinct().collect()
      labels.foreach(println)

      //filter each set of features per the target label (species)
      val labelsIrisRDD: Array[(String, RDD[iris])] = labels.map { label =>
        (label, irisBaseRDD.filter(_.species == label))
      }
      //get the statistics for each RDD identified by a label
      val labelStatsRDD: Array[(String, (StatCounter, StatCounter, StatCounter, StatCounter))] =
        labelsIrisRDD.map { case (label, rdd) =>
          val fstats1: StatCounter = rdd.map(x => x.sepalLength).stats()
          val fstats2: StatCounter = rdd.map(x => x.sepalWidth).stats()
          val fstats3: StatCounter = rdd.map(x => x.petalLength).stats()
          val fstats4: StatCounter = rdd.map(x => x.petalWidth).stats()
          (label, (fstats1, fstats2, fstats3, fstats4))
        }
      println(labelStatsRDD.foreach { case (label, stats) =>
        println(s"target = $label")
        println(stats)
      })
    }


    val baseRDD: RDD[Array[String]] =
      sc.textFile(packagePath + "data/iris.data")
      .map(_.split(",")).filter(x => x.length == 5)

    calculateStatsSpread(baseRDD)
    naiveBayesSparkML(baseRDD)

  }
}