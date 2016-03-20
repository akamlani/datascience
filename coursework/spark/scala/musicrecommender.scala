package part2

import java.nio.file.Paths
import java.sql.Timestamp

import org.apache.spark.rdd.RDD
import org.apache.spark.{Accumulator, SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, SQLContext}

import scala.collection.JavaConverters._
import scala.util.Try


object musicrecommender {
  case class MusicRating(userid: Long, gid: Long, counton: Int)
  case class MusicTab(userid: Long, gid: Long, counton: Int, artistname: String)

  def main(args: Array[String]): Unit = {
    //val conf = new SparkConf().setAppName("Workshop").setMaster("mesos://zk://app200.cluster1:2181/mesos")
    val conf = new SparkConf().setAppName("MusicRec").setMaster("local[*]")
    val sc = new SparkContext(conf)
    println(sc.master)

    val packagePath = "/Users/akamlani/Projects/datascience/packages/spark/data/exercises/package/"
    case class ArtistAlias(badid: Long, goodid: Long)
    case class Artist(artistid: Long, artistname: String)
    case class UserArtist(userid: Long, artistid: Long, playcount: Int)

    def cleanArtistAliasData(): RDD[String] = {
      //Parse and clean the alias RDD format, we don't include the rows that we don't have both correct formatted ids for
      //separator = tab
      val artistAliasBaseRDD: RDD[String]       = sc.textFile(packagePath + "profiledata_06-May-2005/artist_alias.txt")
      val artistAliasCleanLinesRDD: RDD[String] = artistAliasBaseRDD.filter { x =>
        val arr = x.split("\t")
        arr.size == 2 && arr(0).length > 0 && arr(1).length > 0
      }
      artistAliasCleanLinesRDD.saveAsTextFile(packagePath + "/export/artistalias_cleaned")
      artistAliasCleanLinesRDD
    }

    def loadArtistAliasData(): RDD[ArtistAlias]  = {
      val artistAliasLinesReloadRDD: RDD[String] = sc.textFile(packagePath + "/export/artistalias_cleaned")
      val artistAliasLinesRDD: RDD[ArtistAlias] = artistAliasLinesReloadRDD.map(_.split('\t')).map(row =>
        ArtistAlias(badid=row(0).toLong, goodid=row(1).toLong))
      artistAliasLinesRDD
    }

    def cleanArtistData() = {
      //Clean the data and only work off the artist data including the first two columns
      //Clean the data as there are some missing entries in the first column so we cannot convert to type Long
      def parseLong(s: String): Option[Long] = Try(s.toLong).toOption
      //seperator = tab
      val artistDataBaseRDD = sc.textFile(packagePath + "profiledata_06-May-2005/artist_data.txt")
      val colCountRDD = artistDataBaseRDD.map(x => x.split('\t')).map(x => (x.size, 1 )).reduceByKey(_ + _ )
      val artistDataColLimitedRDD = artistDataBaseRDD.filter(x => x.split("\t").size == 2)
      val artistCleanDataRDD = artistDataColLimitedRDD.map(_.split("\t")).filter(row => parseLong(row(0)) != None)
      //Save and reload to File; to save to file it needs to be in string format
      artistCleanDataRDD.map(ac => ac(0).toString + "\t" + ac(1).toString)
        .saveAsTextFile(packagePath + "/export/artist_cleaned")
    }

    def loadCleanedArtistData(): RDD[Artist] = {
      //Load the cleaned Artist information
      val artistCleanDataReloadRDD: RDD[String] = sc.textFile(packagePath + "/export/artist_cleaned")
      //Look at the artist information
      val artistCleanLinesRDD: RDD[Artist] = artistCleanDataReloadRDD.map(_.split("\t")).map(row =>
        Artist(artistid=row(0).toLong, artistname=row(1).toString))
      artistCleanLinesRDD
    }

    def remapArtistData(badIdsRDD: RDD[Long], artistAliasRDD: RDD[ArtistAlias]) = {
      val artistCleanLinesRDD: RDD[Artist] = loadCleanedArtistData()
      val artistAliasKeyMapRDD = artistAliasRDD.map(al =>  (al.badid, al.goodid)).groupByKey()
      val artistKeyMapRDD      = artistCleanLinesRDD.map(ar =>  (ar.artistid, ar.artistname)).groupByKey()
      val artistLinkKeyMapRDD  = artistKeyMapRDD.map(km => (km._1, km._2.toList(0)))
      val artistBadIdsRDD      = artistKeyMapRDD.map(ar => ar._1).intersection(badIdsRDD)
      val artistGoodIdsRDD     = artistKeyMapRDD.map(ar => ar._1).subtract(artistBadIdsRDD)
      val artistGoodRDD        = artistKeyMapRDD.subtractByKey(artistBadIdsRDD.map(k => (k, 1))).map(km => (km._1, km._2.toList(0)))
      val assocArtistIdDataRDD = artistKeyMapRDD.join(artistAliasKeyMapRDD)

      val remappedArtistDataRDD  = assocArtistIdDataRDD.map {case (bid: Long, km) =>
        val v1 = km._1.toList(0)
        val v2 = km._2.toList(0)
        val meta: String = if (  v1.getClass.getName == "java.lang.String") v1 else v2.toString
        val gid:  Long   = if (  v1.getClass.getName == "long") v1.toLong else v2
        (gid, meta)
      }
      val remappedArtistsSetRDD = artistGoodRDD.union(remappedArtistDataRDD)
      remappedArtistsSetRDD.map(ar => ar._1.toString + "\t" + ar._2)
        .saveAsTextFile(packagePath + "/export/artist_cleaned_remapped")
    }

    def loadRemappedArtistData(): RDD[Artist] = {
      //Load the cleaned Artist information
      val artistRemappedReloadRDD: RDD[String] = sc.textFile(packagePath + "/export/artist_cleaned_remapped")
      artistRemappedReloadRDD.map(_.split("\t")).map(row =>
        Artist(artistid=row(0).toLong, artistname=row(1).toString))
    }


    def remapUserArtistData(badIdsRDD: RDD[Long], artistAliasRDD: RDD[ArtistAlias]) = {
      //Load the userArtist information
      //seperator = space
      val userArtistDataBaseRDD   = sc.textFile(packagePath + "profiledata_06-May-2005/user_artist_data.txt")
      val userArtistLinesRDD      = userArtistDataBaseRDD.map(_.split(' ')).map(row =>
        UserArtist(userid=row(0).toLong, artistid=row(1).toLong, playcount=row(2).toInt))

      //create lookup keymaps for the data
      val artistAliasKeyMapRDD = artistAliasRDD.map(al =>  (al.badid, al.goodid)).groupByKey()
      val userArtistKeyMapRDD  = userArtistLinesRDD.map(ua  =>  (ua.artistid, (ua.userid, ua.playcount)))//.groupByKey()

      //Create Transformations to associate the bad keys that need to be replaced based on keymaps
      val userArtistBadIdsRDD = userArtistKeyMapRDD.map(ua => ua._1).intersection(badIdsRDD)
      val userArtistGoodRDD: RDD[(Long, (Long, Int))] = {
        userArtistKeyMapRDD.subtractByKey(userArtistBadIdsRDD.map(k => (k, 1)))
          .map(km => (km._1, km._2/*.toList(0)*/))
      }
      val assocUserArtistIdDataRDD = userArtistKeyMapRDD.join(artistAliasKeyMapRDD)

      //Replace the user artist data (bad ids)
      val remappedUserArtistDataRDD: RDD[(Long, (Long, Int))] = assocUserArtistIdDataRDD.map { case (bid: Long, km ) =>
        val tu: (Long, Int) = km._1/*.toList(0)*/
      val id: Long = km._2.toList(0)
        val meta: (Long,Int) = tu     //if ( tu.getClass.getName.startsWith("scala.Tuple"))  tu
      val gid:  Long = id.toLong      //if (! id.getClass.getName.startsWith("scala.Tuple")) id.toLong
        (gid, meta)
      }
      val remappedUserArtistsSetRDD = userArtistGoodRDD.union(remappedUserArtistDataRDD)
      //format to write to file
      remappedUserArtistsSetRDD.map(ua => ua._2._1.toString + "\t" + ua._1 + "\t" + ua._2._2)
        .saveAsTextFile(packagePath + "/export/userartist_remapped")
    }

    def createRatings(): RDD[MusicRating] = {
      //load from checkpoint of cleaned and remapped userartist data
      val remappedReloadedUserArtistRDD: RDD[Array[String]] = sc.textFile(packagePath + "/export/userartist_remapped")
        .map(_.split('\t'))
      //create rating class
      remappedReloadedUserArtistRDD.map(ua => MusicRating(userid=ua(0).toLong, gid=ua(1).toLong, counton=ua(2).toInt))
    }

    //DEBUG CNTS for userArtist
    //println(artistAliasKeyMapRDD.count())             //190892
    //println(userArtistLinesRDD.count())               //24296858
    //println(userArtistKeyMapRDD.count())              //1631028;    24296858
    //println(userArtistBadIdsRDD.count())              //62931;      62931
    //println(userArtistGoodRDD.count())                //1568097;    23878789
    //println(assocUserArtistIdDataRDD.count())         //62931;      418069
    //println(remappedUserArtistDataRDD.count())        //62931;      418069
    //println(remappedUserArtistsSetRDD.count())        //1631028;    1247161057  ---> <TBD>, 24296858
    //println(musicRatingsBaseRDD.count())              //(TBD);      TBD         ---> <TBD>, 24296858

    //DEBUG CNTS for Artist
    //println(artistAliasKeyMapRDD.count())             //190892
    //println(artistCleanLinesRDD.count())              //1848172
    //println(artistKeyMapRDD.count())                  //1848172
    //println(artistBadIdsRDD.count())                  //189842
    //println(artistGoodRDD.count())                    //1658330
    //println(assocArtistIdDataRDD.count())             //189842
    //println(remappedArtistDataRDD.count())            //189842
    //println(remappedArtistsSetRDD.count())            //18488172
    //remappedArtistDataRDD.take(10).foreach(println)
    //remappedArtistsSetRDD.take(10).foreach(println)

    //joint: 24282051
    //leftouterjoin: 24296858
    //music ratings - bad ids: 20
    //music ratings - artiskeymap: 14807

    //**********Begin User Artist Sequence***********
    //val artistAliasCleanedLinesRDD: RDD[String] = cleanArtistAliasData()
    val artistAliasLinesRDD: RDD[ArtistAlias]   = loadArtistAliasData()
    val badIdsRDD: RDD[Long] = artistAliasLinesRDD.map(al => al.badid)

    //remapUserArtistData(badIdsRDD, artistAliasLinesRDD)
    val musicRatingsBaseRDD: RDD[MusicRating] = createRatings()
    val sqlContext: SQLContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._

    //musicRatingsBaseRDD.take(5).foreach(println)
    val musicRatingsDF: DataFrame = musicRatingsBaseRDD.map(mr => MusicRating(mr.userid, mr.gid, mr.counton)).toDF()
    musicRatingsDF.registerTempTable("musicrating")
    //sqlContext.sql("SELECT userid, counton from musicrating where counton > 3 LIMIT 5").collect.foreach(println)

    //**********Begin Artist Sequence***********
    //cleanArtistData()
    //loadCleanedArtistData()
    //remapArtistData(badIdsRDD, artistAliasLinesRDD)
    val artistReloadBaseRDD = loadRemappedArtistData()
    //println(artistReloadBaseRDD.count())
    //artistReloadBaseRDD.take(10).foreach(println)

    //**********Add to the MusicRatings Information *****
    val artistKeyMapRDD: RDD[(Long, Iterable[String])] = artistReloadBaseRDD.map(ar => (ar.artistid, ar.artistname)).groupByKey()
    val musicRatingTableRDD: RDD[MusicTab] = musicRatingsBaseRDD.map(mr => (mr.gid, (mr.userid, mr.counton)))
      .leftOuterJoin(artistKeyMapRDD)
      .mapValues(f = kv => (kv._1._1, kv._1._2, kv._2.map(a => a.toList(0)).getOrElse("") ) )
      .map(kv => MusicTab(userid=kv._2._1, gid=kv._1, counton=kv._2._2, artistname=kv._2._3))
    val musicRatingTableDF: DataFrame = musicRatingTableRDD.toDF()
    musicRatingTableDF.registerTempTable("musicratingfull")
    musicRatingTableDF.printSchema()
    musicRatingTableDF.explain(true)
    sqlContext.sql("SELECT userid, gid, artistname from musicratingfull where counton > 3 LIMIT 5").collect.foreach(println)

    //println( musicRatingsBaseRDD.map(mr => mr.gid).intersection(badIdsRDD).count() )
    //println( musicRatingsBaseRDD.map(mr => mr.gid).subtract(artistKeyMap.keys).count() )
    //musicRatingTableRDD.take(10).foreach(println)
    //println( musicRatingTableRDD.count)

/*
    //ARCHIVED CODE
    def parseString(s: String): Option[String] = Try(s.toString).toOption
    val remappedArtistsSetRDD = remappedArtistDataRDD.leftOuterJoin(artistGoodRDD)
    val flattenRemappedArtistDataRDD = remappedArtistsSetRDD.groupByKey().mapValues{ case ar =>
      ar.flatMap( tu => List( parseString(tu._1), parseString(tu._2.get)))
    }.flatMapValues(data => data).map(ar => (ar._1, ar._2.get)).cache()
*/

  }
}