
package xikai.minesweeper

object Driver {

  def startGUI(
    weightsFilename: String = "net_border2_boardSize16.zip", // Saved NN weights
    bSize: Int = 16,
    args: Array[String] = Array()
  ): Unit = {
    val aiPlayer = QNNPlayer.toPlayer(QNNPlayer.loadQNNPlayer(bSize, weightsFilename))
    val gui = new GameGUI(bSize, bSize, 40, 40, aiPlayer)
    gui.main(args)
    println("exiting")
    sys.exit(0)
  }

  def trainNewNN(
    saveFilename: String = "net_border2_boardSize16.zip",
    boardSize: Int = 16,
    rounds: Int = 200,
    loadFromDisk: Boolean = false
  ): Unit = {
    //val nn = NNPlayer.untrainedPlayer(boardSize=boardSize, seed=1, iter=1, learningRate=0.00003) // works for 16 x 16
    val nn = if (loadFromDisk) {
      NNPlayer.loadNNPlayer(boardSize, saveFilename)
    } else {
      NNPlayer.untrainedPlayer(boardSize=boardSize, seed=1, iter=1, learningRate=0.001)
    }

    for (i <- 0 until rounds) {
      val (trainingInput, trainingOutput) = nn.createTrainingData(100)
      //println("Avg error before: " + nn.avgErrorOnData(trainingInput, trainingOutput))
      nn.train(trainingInput, trainingOutput, batchSize=1, epochs=1)
      
      //println("Avg error after: " + nn.avgErrorOnData(trainingInput, trainingOutput))
      if (i % 50 == 0) {
        println(Player.playSeries(boardSize=boardSize, Player.randomMovePlayer,
                                             NNPlayer.toPlayer(nn), 1000))
      }

      if (i == rounds - 1) {
        nn.saveNetwork(saveFilename)
        nn.playAgainstSelf(debug=false).foreach { gs =>
          println(gs + "\n\n")
        }
      }
    }
  }

  def trainNewQNN(
    loadFilename: String = "",
    saveFilename: String = "net_border2_boardSize16.zip",
    boardSize: Int = 16,
    rounds: Int = 200,
    loadFromDisk: Boolean = false,
    learningRate: Option[Double] = None
  ): Unit = {
    //val nn = NNPlayer.untrainedPlayer(boardSize=boardSize, seed=1, iter=1, learningRate=0.00003) // works for 16 x 16
    val nn = if (loadFromDisk) {
      QNNPlayer.loadQNNPlayer(boardSize, loadFilename)
    } else {
      QNNPlayer.untrainedPlayer(boardSize=boardSize, seed=1, learningRate=0.005)
    }
    
    learningRate match {
      case Some(rate) => nn.setLearningRate(rate)
      case _ =>
    }


    //val nn2 = QNNPlayer.loadQNNPlayer(boardSize, loadFilename + "_morning")


    for (i <- 0 until rounds) {
      val (trainingInput, trainingOutput) = nn.createTrainingData(100)
      println("Avg error before: " + nn.avgErrorOnData(trainingInput, trainingOutput))
      nn.train(trainingInput, trainingOutput, batchSize=1, epochs=1)
      println("Avg error after: " + nn.avgErrorOnData(trainingInput, trainingOutput))
      if (i % 20 == 0) {
        println(Player.playSeriesScore(boardSize=boardSize, Player.randomMovePlayer,
          QNNPlayer.toPlayer(nn), 200))
        nn.saveNetwork(saveFilename)
      }

      if (i == rounds - 1) {
        nn.saveNetwork(saveFilename)
        nn.playAgainstSelf(debug=false).foreach { gs =>
          println(gs)
          println("blank line")
        }
        println("done")
      }
    }
  }

  def experiment() = {
    val qnn = QNNPlayer.untrainedPlayer(boardSize=5, seed=1, learningRate=0.4)
    val game = GameState.createNewGame(5)
    val input = QNNPlayer.gameStateToNNInput(game)
    val output = QNNPlayer.gameStateToNNOutput(game)
    println("Output goal:")
    println(output.toVector)
    println("\nCurrent output:")
    println(qnn.rawOutput(game))
    import scala.util.Random;
    for (i <- 0 to 3000) {
      if (Random.nextInt(5) == 0) {
        output(0) = 1.0
      } else {
        output(0) = 0.0
      }
      qnn.train(Array.fill(1)(input), Array.fill(1)(output), 1, 1)
      
      //qnn.train(Array.fill(5)(input), Array.fill(5)(output), 1, 1)
    }
    println("\nUpdated output:")
    println(qnn.rawOutput(game))
  }

  def main(args: Array[String]): Unit = {
    //startGUI(args = args)

    //trainNewNN("nn2_bsize_6", 6, loadFromDisk=true)
    //trainNewQNN("nn_bsize_4", 4, loadFromDisk=false)
    
    startGUI("qnn7_bsize_16_gamma99_3", 16)
    sys.exit(0)

    // For two hidden layers n=8, lr = 0.01 worked
    trainNewQNN(loadFilename="dense8_pool", "dense8_pool", 8, rounds=600, loadFromDisk=false, learningRate=Some(0.02))
    //val nn = QNNPlayer.loadQNNPlayer(8, "dense8_pool")
    //startGUI("dense8", 8)
    sys.exit(0)
    //val n = 16
    //val nn = NNPlayer.loadNNPlayer(n, "net_border2_boardSize16.zip") //NNPlayer.untrainedPlayer(n, 12, 1, 0.00003)
    //experiment()

    // qnn_bsize_16 was 2 iterations with learningRate = 0.05 (maybe too high? didn't appear to learn)
    // for n=5, will train with learning rate 0.1, then drop to 0.05 later. Start with gamma=0, then increase.
    // for n=10, lr = 0.1, 1 round. (0.01?)
    // for n=16, lr = 0.005, a few rounds, then moved to 0.004. 4 layers + output layer, nOut = 20 for all.
    val n = 16
    val nnfname = s"qnn8_bsize_${n}"
    //val nnfname = "qnn7_bsize_16_gamma99"
    /*
    val nn = QNNPlayer.loadQNNPlayer(n, nnfname)
    val nn2 = QNNPlayer.loadQNNPlayer(n, nnfname + "_morning")

    for (i <- 0 to 10) {
      println(Player.playSeriesScore(boardSize=n, QNNPlayer.toPlayer(nn2),
        QNNPlayer.toPlayer(nn), 100))
    }*/
    
    //startGUI(nnfname, 16)

    //sys.exit(0)
    //trainNewQNN(loadFilename=nnfname, nnfname, n, rounds=10000, loadFromDisk=true, learningRate=Some(0.004))
    //sys.exit(0)
    //val nn = QNNPlayer.loadQNNPlayer(n, nnfname)
    val nn = QNNPlayer.untrainedPlayer(boardSize=n, seed=1, learningRate=0.1)
    /*
    import org.deeplearning4j.nn.conf.GradientNormalization;
    import org.deeplearning4j.nn.conf.NeuralNetConfiguration
    val conf = nn.neuralNetwork.conf()
    val modconf = new NeuralNetConfiguration.Builder(conf)
          .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
      .gradientNormalizationThreshold(1.0)
      .build
    //println(conf)
    //println(modconf)
    nn.neuralNetwork.setConf(modconf)
     */
    //sys.exit(0)
    nn.setLearningRate(0.004)
    nn.erTrain(nnfname)
    nn.saveNetwork(nnfname)
    //println("rnd vs rnd: " + Player.playSeries(boardSize=n, Player.randomMovePlayer, Player.randomMovePlayer, 1000))
    //println(Player.playSeries(boardSize=n, Player.randomMovePlayer, QNNPlayer.toPlayer(nn), 200)*5)
     

    // non-q-learning version
    
    /*
    val n = 5
    trainNewNN(s"nn_bsize_${n}", n, rounds=200, loadFromDisk=false)
    val nn = NNPlayer.loadNNPlayer(n, s"nn_bsize_${n}")
    println("rnd vs rnd: " + Player.playSeries(boardSize=n, Player.randomMovePlayer, Player.randomMovePlayer, 1000))
    println(Player.playSeries(boardSize=n, Player.randomMovePlayer,
      NNPlayer.toPlayer(nn), 1000))
     */

    //println(nn.neuralNetwork.paramTable().get("4_W").shapeInfoToString())
    /*
    import org.nd4j.linalg.factory.Nd4j
    val (trainingInput, trainingOutput) = nn.createTrainingData(20)
    println(nn.neuralNetwork.output(Nd4j.create(trainingInput(10))))
    println(trainingOutput(10).toVector)
     */

    //val (trainingInput, trainingOutput) = nn.createTrainingData(100)
    /*
    println("rnd vs rnd: " + Player.playSeries(boardSize=n, Player.randomMovePlayer, Player.randomMovePlayer, 1000))

    for (i <- 0 until 10000) {
      val (trainingInput, trainingOutput) = nn.createTrainingData(300)
      //println("Avg error before: " + nn.avgErrorOnData(trainingInput, trainingOutput))
      nn.train(trainingInput, trainingOutput, batchSize=1, epochs=1)
      
      //println("Avg error after: " + nn.avgErrorOnData(trainingInput, trainingOutput))
      if (i % 200 == 0)
      println(Player.playSeries(boardSize=n, Player.randomMovePlayer, NNPlayer.toPlayer(nn), 1000))
      if (i % 500 == 499) {
        nn.saveNetwork("net_border2_boardSize16.zip")
        nn.playAgainstSelf(debug=false).foreach { gs =>
          println(gs + "\n\n")
        }
      }
    }*/

    /*
    val out = nn.neuralNetwork.output(Nd4j.create(trainingInput(10)))
    val actualOut = trainingOutput(10).toVector
    for (i <- 0 until actualOut.length) {
      println(out.getDouble(i) + " , " + actualOut(i))
    }*/

    //println(nn.neuralNetwork.output(Nd4j.create(trainingInput(10))))
    //println(trainingOutput(10).toVector)

    /*
    nn.playAgainstSelf(debug=true).foreach { gs =>
      println(gs + "\n\n")
    }*/
     
    //println(Player.playSeries(boardSize=n, Player.randomMovePlayer, NNPlayer.toPlayer(nn), 10000))
    //println(Player.playSeries(boardSize=n, Player.randomMovePlayer, Player.randomMovePlayer, 10000))
    //println(Player.playSeries(boardSize=n, Player.randomMovePlayer, NNPlayer.toPlayer(nn), 1))

    //println(Player.playSeries(boardSize=6, Player.randomMovePlayer, Player.randomMovePlayer, 10000))
    /*
    val gui = new GameGUI(16, 16, 30, 30)
    gui.main(args)
    println("exiting")
     */
  }
}
