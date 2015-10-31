


../../spark-1.4.1-bin-hadoop2.6/bin/spark-submit \
--class "com.databricks.apps.twitter_classifier.Collect"
--master "spark://ukko160:7077"\
target/scala-2.10/spark-twitter-lang-classifier-assembly-1.0.jar\
./tmp/tweets\
1000\
10\
1\
--consumerKey RwdOf9hFD9YuV6diy4BgndUFD\
--consumerSecret a6RQ4s8hCGSTNYNe4tE1cep6Ospsm7p3FIC8vE6VAz4A3RX20M\
--accessToken 169986084-tATCsFlagD7ONl8D98ZBKaTiqtFnuXWZDi7qHH21\
--accessTokenSecret lJIfrn1U5bW1r4Mcyhz46TGpUEZawacEV4VhUyThMiLNH
