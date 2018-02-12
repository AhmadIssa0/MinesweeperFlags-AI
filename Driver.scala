
package xikai.minesweeper

object Driver {

  def startGUI(
    weightsFilename: String = "net_border2_boardSize16.zip", // Saved NN weights
    bSize: Int = 16,
    args: Array[String] = Array()
  ): Unit = {
    val aiPlayer = NNPlayer.toPlayer(NNPlayer.loadNNPlayer(bSize, weightsFilename))
    val gui = new GameGUI(bSize, bSize, 30, 30, aiPlayer)
    gui.main(args)
    println("exiting")
    sys.exit(0)
  }

  def trainNewNN(
    saveFilename: String = "net_border2_boardSize16.zip",
    boardSize: Int = 16,
    rounds: Int = 100
  ): Unit = {
    val nn = NNPlayer.untrainedPlayer(boardSize=boardSize, seed=1, iter=1, learningRate=0.00003)

    for (i <- 0 until rounds) {
      val (trainingInput, trainingOutput) = nn.createTrainingData(300)
      //println("Avg error before: " + nn.avgErrorOnData(trainingInput, trainingOutput))
      nn.train(trainingInput, trainingOutput, batchSize=1, epochs=1)
      
      //println("Avg error after: " + nn.avgErrorOnData(trainingInput, trainingOutput))
      if (i % 50 == 0) {
        println(Player.playSeries(boardSize=boardSize, Player.randomMovePlayer,
                                             NNPlayer.toPlayer(nn), 1000))
      }

      if (i % 100 == 99) {
        nn.saveNetwork(saveFilename)
        nn.playAgainstSelf(debug=false).foreach { gs =>
          println(gs + "\n\n")
        }
      }
    }
  }


  def main(args: Array[String]): Unit = {
    startGUI(args = args)
    //trainNewNN("test_bsize_6", 6)
    //startGUI("test_bsize_6", 6)
    //val n = 16
    //val nn = NNPlayer.loadNNPlayer(n, "net_border2_boardSize16.zip") //NNPlayer.untrainedPlayer(n, 12, 1, 0.00003)
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
