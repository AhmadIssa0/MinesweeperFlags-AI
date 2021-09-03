
package xikai.minesweeper

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, DenseLayer, OutputLayer, SubsamplingLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.ConvolutionMode
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.NesterovsUpdater;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.nd4j.linalg.learning.config.Adam

import java.io.File;

// Needed for stats to analyse the neural network.
/*
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.api.storage.StatsStorage
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
 */


/* Problems:
 * 
 * 
 * 
 */

case class QNNPlayer(boardSize: Int, neuralNetwork: MultiLayerNetwork) {
  val nChannels = neuralNetwork.layerInputSize(0) // number of channels in input of CNN

  // Needed to analyse the neural network.
  /*
  val uiServer: UIServer = UIServer.getInstance() // For deeplearning4j network visualization
  val statsStorage: StatsStorage = new InMemoryStatsStorage()
  val listenerFrequency = 1
  neuralNetwork.setListeners(new StatsListener(statsStorage, listenerFrequency))
  uiServer.attach(statsStorage)
   */
  //Finally: open your browser and go to http://localhost:9000/train

  override def clone() = QNNPlayer(boardSize, neuralNetwork.clone())

  def setLearningRate(learningRate: Double) = {
    neuralNetwork.setLearningRate(learningRate)
  }

  def train(input: Array[Array[Double]],
    label: Array[Array[Double]],
    batchSize: Int,
    epochs: Int
  ): Unit = {
    neuralNetwork.setInputMiniBatchSize(batchSize)

    val inputTrain:INDArray = Nd4j.create(input)
    val labelTrain:INDArray = Nd4j.create(label)

    val ds = new DataSet(inputTrain, labelTrain)
    ds.shuffle()
    //val dsIter: DataSetIterator = ds.iterateWithMiniBatches()

    //println("Train model...")
    for (i <- 0 until epochs) {
      neuralNetwork.fit(new ListDataSetIterator(ds.batchBy(batchSize)))
    }
  }
  
  // filename should be a .zip file.
  def saveNetwork(filename: String) = {
    val locationToSave = new File(filename) //Where to save the network. Note: the file is in .zip format - can be opened externally
    val saveUpdater = true //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
    ModelSerializer.writeModel(neuralNetwork, locationToSave, saveUpdater);
  }

  def avgErrorOnData(inputData: Array[Array[Double]],
    outputData: Array[Array[Double]]): Double = {
    var error: Double = 0.0
    for (i <- 0 until inputData.length) {
      val out: INDArray = rawOutput(inputData(i))
      val n = outputData(i).length
      for (j <- 0 until n) {
        val a = (outputData(i)(j) - out.getDouble(j.asInstanceOf[java.lang.Integer]))
        error += a*a/n
      }
    }
    error/inputData.length // mean squared error
  }

  def output(gs: GameState): Vector[Vector[Double]] = {
    val n = gs.boardSize
    val out: INDArray = neuralNetwork.output(
      Nd4j.create(QNNPlayer.gameStateToNNInput(gs), Array[Int](1,nChannels*boardSize*boardSize), 'c'))
    // 5 is the number of channels. Input needs to be a column vector. Hence we specify the shape.

    for (j <- (0 until n).toVector) yield {
      for (i <- (0 until n).toVector) yield {
        out.getDouble((j*n + i).asInstanceOf[java.lang.Integer])
      }
    }
  }

  def rawOutput(inputData: Array[Double]): INDArray = {
     val out: INDArray = neuralNetwork.output(
      Nd4j.create(inputData, Array[Int](1,nChannels*boardSize*boardSize), 'c'))
    out
  }

  def rawOutput(gs: GameState): INDArray = {
    rawOutput(QNNPlayer.gameStateToNNInput(gs))
  }


  def rewardOfBestMove(gs: GameState): Double = {
    if (!gs.isBluesTurn) rewardOfBestMove(gs.reverseRedAndBlue)
    else if (gs.isGameOver) 0.0
    else {
      assert(gs.boardSize == boardSize,
        s"NNPlayer trained on size $boardSize, not ${gs.boardSize}")
      val n = boardSize
      val output: Vector[Vector[Double]] = this.output(gs)

      var largestOutput: Double = 0.0
      for (j <- (0 until n);
        i <- (0 until n)) {
        if (!gs.visible(i)(j)) {
          val out = output(j)(i)
          if (out >= largestOutput) {
            largestOutput = out
          }
        }
      }
      largestOutput
    }
  }

  def bestMove(gs: GameState, debug: Boolean = false): Option[Move] = {
    if (gs.isGameOver) None
    else if (!gs.isBluesTurn) bestMove(gs.reverseRedAndBlue, debug)
    else {
      assert(gs.boardSize == boardSize,
        s"QNNPlayer trained on size $boardSize, not ${gs.boardSize}")
      val n = boardSize
      val output: Vector[Vector[Double]] = this.output(gs)
      //println(output.map(_ mkString ", ").mkString("\n"))
      if (debug) { 
        println(output)
        println("Visible:")
        println(gs.visible)
      }

      //var unset = true // No valid move set yet
      var largestOutput: Double = -100000.0 // Should be -inf
      var (iOfLargest, jOfLargest) = (-1, -1)
      for (j <- (0 until n);
        i <- (0 until n)) {
        if (!gs.visible(i)(j)) {
          val out = output(j)(i)
          if (out >= largestOutput) {
            largestOutput = out
            iOfLargest = i
            jOfLargest = j
            //unset = false
          }
        }
      }
      //println(s"NN best move: ${(iOfLargest, jOfLargest)}")
      assert((iOfLargest, jOfLargest) != (-1, -1), s"NN played invalid move: $gs\n$output")
      Some((iOfLargest, jOfLargest))
    }
  }

  /**  
    * Return all positions from a game against itself.
    */
  def playAgainstSelf(debug: Boolean = false): Vector[GameState] = {
    def gamePositions(currGS: GameState, allPositions: Vector[GameState]): Vector[GameState] = {
      if (currGS.isGameOver) allPositions
      else {
        val move = bestMove(currGS, debug).get
        val nextGS: GameState = currGS.makeMove(move._1, move._2) match {
          case Right(gs) => gs
          case Left(error) =>
            throw new Exception(s"AI tries to play move: $move. But received error: $error")
        }
        gamePositions(nextGS, allPositions :+ nextGS)
      }
    }

    val gs = GameState.createNewGame(boardSize)
    gamePositions(gs, Vector(gs))
  }

  def epGreedyMove(gs: GameState, ep: Double): Option[Move] = {
    // ep is the probability we play a random move instead of a best move.
    import scala.util.Random;
    if (gs.isGameOver) None
    else if (Random.nextDouble() < ep)
      RandomMovePlayer.bestMove(gs)
    else bestMove(gs)
  }

  // Experience replay train
  /*
   * ep - Probability of selecting random move.
   */
  def erTrain(saveFilename: String,
    iterations: Int = 10000,
    bufferSize: Int = 5000,
    batchSize: Int=1,
    ep: Double = 0.3, // probability of making random move
    freezeIterations: Int = 100 // every this many iterations update the targetNetwork
  ) = {
    var buffer = new ERBuffer[Experience](bufferSize)
    var game = GameState.createNewGame(boardSize)
    var targetNetwork: QNNPlayer = clone()


    val nn = QNNPlayer.loadQNNPlayer(boardSize, "qnn7_bsize_16_gamma99")

    def refill() = {
      buffer = new ERBuffer[Experience](bufferSize)
      while (!buffer.isFilled) {
        if (game.isGameOver)
          game = GameState.createNewGame(boardSize)

        val stateBefore = game
        val move = epGreedyMove(game, ep).get // We're using this network to play
        game = game.makeMove(move._1, move._2) match {
          case Right(gs) => gs
          case Left(error) =>
            throw new Exception(s"AI tries to play move: $move. But received error: $error")
        }

        buffer.add((stateBefore, bestMove(stateBefore).get))
      }
      println("buffer filled")
    }

    
    game = GameState.createNewGame(boardSize)
    refill()

    for (i <- 0 until iterations) {
      if (i % 6*4 == 0) {
        println(Player.playSeries(boardSize=boardSize, Player.randomMovePlayer, QNNPlayer.toPlayer(this), 100))
        //println(Player.playSeriesScore(boardSize=boardSize, QNNPlayer.toPlayer(nn), QNNPlayer.toPlayer(this), 100))    
      }

      if (i % freezeIterations == 0) {
        targetNetwork = clone()
        println("updated network")
      }

      if (i % freezeIterations == 0) {
        saveNetwork(saveFilename)
        println("saved network")
      }

      for (j <- 0 until batchSize) {
        if (game.isGameOver)
          game = GameState.createNewGame(boardSize)

        val stateBefore = game
        val move = epGreedyMove(game, ep).get // We're using this network to play
        game = game.makeMove(move._1, move._2) match {
          case Right(gs) => gs
          case Left(error) =>
            throw new Exception(s"AI tries to play move: $move. But received error: $error")
        }

        if (!game.isGameOver)
          buffer.add((stateBefore, bestMove(stateBefore).get)) // the move to store is the move by the QNN not the policy
      }

      trainingStep(buffer.batch(batchSize), targetNetwork)
    }
  }

  def trainingStep(experiences: Vector[Experience], targetNetwork: QNNPlayer) = {
    val input: Array[Array[Double]] = experiences.map(e => QNNPlayer.gameStateToNNInput(e._1)).toArray


    val alpha: Double = 0.9 // learning rate
    val gamma: Double = 0//0.99 // discount factor
        // Flattened board with updated neural network output values
    def updatedNNOutput(gs: GameState, move: Move): Array[Double] = {
      if (! gs.isBluesTurn) updatedNNOutput(gs.reverseRedAndBlue, move) // The NN plays from blue player perspective
      else {
        val currOutput = this.output(gs)
        val n = gs.boardSize
        val output = new Array[Double](n * n)
        
        //val raw = this.rawOutput(gs)
        //val output = (0 until n*n).toArray.map(x => raw.getDouble(x.asInstanceOf[java.lang.Integer]))
        /*
        for (j <- 0 until n;
          i <- 0 until n) {
          val index = j*n + i
          output(index) = currOutput(i)(j)
        }

        val (i, j) = move // not using move atm
         */


        for (j <- 0 until n;
          i <- 0 until n) {
          val index = j*n + i // this is this.output(j)(i) which on the board is board(i)(j)
          if (gs.visible(i)(j)) {
            output(index) = 0.0 // Penalize playing invalid move (e.g. autolose game)
                                 //output(index) = if (gs.board(i)(j) == FLAG) 1.0 else 0.0
          } else {
            // Check if it's a game winning move.
            if (gs.board(i)(j) == FLAG && gs.blueScore + 1 >= gs.winningScore) {
              output(index) = (1-alpha)*currOutput(i)(j) + alpha * 1.0 // We won.
            } else {
              val gsNext = gs.makeMove(i, j) match {
                case Right(g) => g
                case Left(error) => throw new IllegalStateException(s"Should not reach here: $error")
              }              
              if (gs.board(i)(j) == FLAG) {
                output(index) = 1.0
                //output(index) = (1-alpha)*currOutput(i)(j) + alpha * (1.0 + gamma * targetNetwork.rewardOfBestMove(gsNext))
              } else {
                //output(index) = (1-alpha)*currOutput(i)(j) + alpha * gamma*(-1.0)*targetNetwork.rewardOfBestMove(gsNext)
                output(index) = 0.0
              }
            }
          }
        }
        
        output
       }
    }

    val targets =
      for ((gs, move) <- experiences) yield {
        updatedNNOutput(gs, move)
      }

    val output = targets.toArray
    //println("Avg error before: " + avgErrorOnData(input, output))
    train(input, output, batchSize=output.length, epochs=1)
    //println("Avg error after: " + avgErrorOnData(input, output))
  }

  /**  Make this NN play against itself to generate training data.
    *  Returns a pair of array of inputs and array of outputs.
    *  @param size Number of game positions to generate training data from.
    */
  def createTrainingData(size: Int): (Array[Array[Double]], Array[Array[Double]]) = {
    //val inputs = Array.ofDim[Double](size, 11*boardSize*boardSize)
    //val outputs = Array.ofDim[Double](size, boardSize*boardSize)

    def positions(n: Int, pos: Vector[GameState]): Vector[GameState] = {
      if (n <= 0) pos
      else {
        //val posInSingleGame = playAgainstSelf()
        //val posInSingleGame = Player.playSingleGame(boardSize, Player.randomMovePlayer, QNNPlayer.toPlayer(this))
        //val posInSingleGame = Player.playSingleGame(boardSize, QNNPlayer.toPlayer(this), QNNPlayer.toPlayer(this))
        val gp = EpGreedyPlayer.toPlayer(EpGreedyPlayer(this, ep=0.1))
        val posInSingleGame = Player.playSingleGame(boardSize, gp, gp)
        // Drop last position, since the end position has no further moves.
        positions(
          n - (posInSingleGame.length - 2),
          pos ++ posInSingleGame.slice(0,math.min(size, posInSingleGame.length-1))
        )
      }
    }

    val games = positions(size, Vector[GameState]())
    val inputs = games.map(gs => QNNPlayer.gameStateToNNInput(gs)).toArray

    val alpha: Double = 0.9
    val gamma: Double = 0.99 // discount factor

    // Flattened board with updated neural network output values
    def updatedNNOutput(gs: GameState): Array[Double] = {
      if (! gs.isBluesTurn) updatedNNOutput(gs.reverseRedAndBlue) // The NN plays from blue player perspective
      else {
        val currOutput = this.output(gs)
        val n = gs.boardSize
        val output = new Array[Double](n * n)
        for (j <- 0 until n;
          i <- 0 until n) {
          val index = j*n + i
          if (gs.visible(i)(j)) {
            output(index) = 0.0 // Penalize playing invalid move (e.g. autolose game)
            //output(index) = if (gs.board(i)(j) == FLAG) 1.0 else 0.0
          } else {
            // Check if it's a game winning move.
            if (gs.board(i)(j) == FLAG && gs.blueScore + 1 >= gs.winningScore) {
              //(1-alpha)*currOutput(x)(y) + alpha * 1.0
              output(index) = 1.0 // We won.
            } else {
              val gsNext = gs.makeMove(i, j) match {
                case Right(g) => g
                case Left(error) => throw new IllegalStateException(s"Should not reach here: $error")
              }
              if (gs.board(i)(j) == FLAG) {
                output(index) = 1.0
                //output(index) = (1-alpha)*currOutput(i)(j) + alpha * (1.0 + gamma * 1.0 * rewardOfBestMove(gsNext))
              } else {
                //output(index) = (1-alpha)*currOutput(i)(j) + alpha * gamma*(-1.0)*rewardOfBestMove(gsNext)
                output(index) = 0.0
              }
            }
          }
        }
        output
       }
    }

    /*
    import scala.collection.parallel._
    val gamesPar = games.par
    val numOfThreads = 6
    gamesPar.tasksupport = new ForkJoinTaskSupport(new scala.concurrent.forkjoin.ForkJoinPool(numOfThreads))
     */
    val outputs = games.map(gs => updatedNNOutput(gs)).toArray
    (inputs, outputs)
  }
}

object QNNPlayer {
  def untrainedPlayer(boardSize: Int, 
                      seed: Int, 
    learningRate: Double,
    momentum: Double=0.9
  ): QNNPlayer = {
    val nChannels: Int = 3
    val inputNum: Int = nChannels*boardSize*boardSize
    val outputNum: Int = boardSize*boardSize
    

    val conf = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
      .gradientNormalizationThreshold(1.0)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    //.optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
      .updater(new Nesterovs(learningRate, momentum))
      //.updater(new Adam(learningRate))
      .l2(learningRate*.01)
      //.weightInit(WeightInit.ZERO)
      .list()
      .layer(new ConvolutionLayer.Builder(3,3)
        .nIn(nChannels)
        .stride(1,1)
        .nOut(20) // 200 // 20
        .convolutionMode(ConvolutionMode.Same)
        .activation(Activation.RELU )
        .build())
    
      .layer(new ConvolutionLayer.Builder(3,3)
        .stride(1,1)
        .nOut(50) // 200 // 20
        .convolutionMode(ConvolutionMode.Same)
        //.activation(Activation.LEAKYRELU )
        .activation(Activation.RELU )
        .build())
    /*
      .layer(new ConvolutionLayer.Builder(3,3)
        .stride(1,1)
        .nOut(20) // 200 // 20
        .convolutionMode(ConvolutionMode.Same)
        .activation(Activation.RELU)
        .build)
    
      .layer(new ConvolutionLayer.Builder(3,3)
        .stride(1,1)
        .nOut(20) // 100 // 30
        .convolutionMode(ConvolutionMode.Same)
        .activation(Activation.RELU)
        .build)
    
      .layer(new ConvolutionLayer.Builder(3,3)
        .stride(1,1)
        .nOut(20) // 50
        .convolutionMode(ConvolutionMode.Same)
        .activation(Activation.RELU)
        .build)

      .layer(new ConvolutionLayer.Builder(3,3)
        .stride(1,1)
        .nOut(20) // 10 // 2
        .convolutionMode(ConvolutionMode.Same)
        .activation(Activation.RELU)
        .build)
     */
      
    /*
       .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
        .kernelSize(2,2)
        .stride(2,2)
         .build())
     */
      .layer(new ConvolutionLayer.Builder(3,3)
        .stride(1,1)
        .nOut(100) // 200 // 20
        .convolutionMode(ConvolutionMode.Same)
        //.activation(Activation.LEAKYRELU )
        .activation(Activation.RELU )
        .build())

      //.layer(new DenseLayer.Builder() //create the first input layer.
      //              .nOut(128)
      //              .build())
      .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
        .nOut(outputNum)
        .activation(Activation.IDENTITY)
        .build())

      .setInputType(InputType.convolutionalFlat(boardSize, boardSize, nChannels))
      .build

    val model = new MultiLayerNetwork(conf)
    model.init()
    //model.setListeners(new ScoreIterationListener(5))

    QNNPlayer(boardSize, model)
  }

  def gameStateToNNInput(gs: GameState): Array[Double] = {
    val n = gs.boardSize
    val input = new Array[Double](3 * n * n)

    for (j <- 0 until n;
         i <- 0 until n) {
      val index = j*n + i
      input(index + 0) = if (gs.visible(i)(j)) 1.0 else 0.0
      input(index + 1*n*n) = if (gs.visible(i)(j) && gs.board(i)(j) == FLAG) {
        if (gs.blueFlagsCaptured.contains((j, i))) 1.0 else -1.0
        // blue is first player
        //1.0
      } else {
        0.0
      }

      if (gs.visible(i)(j) && gs.board(i)(j) != FLAG) {
        //input(index + 2 + gs.board(i)(j)) = 1.0
        input(index + 2*n*n) = gs.board(i)(j) / 8.0
      }

      /*
      if ((i == 0 && j == 0) || (i == n-1 && j == 0) ||
        (i == 0 && j == n-1) || (i == n-1 && j == n-1)) { // Corner cell
        input(index + 3*n*n) = 1.0
      } else if (i == 0 || i == n-1 || j == 0 || j == n-1) { // Border
        input(index + 4*n*n) = 1.0
      }*/

      
      //input(index + 5*n*n) = if (gs.blueFlagsCaptured.contains((i,j))) 1.0 else 0.0
      //input(index + 6*n*n) = if (gs.redFlagsCaptured.contains((i,j))) 1.0 else 0.0
    }

    input
  }

  def gameStateToNNOutput(gs: GameState): Array[Double] = {
    val n = gs.boardSize
    val output = new Array[Double](n * n)

    for (j <- 0 until n;
         i <- 0 until n) {
      val index = j*n + i
      output(index) = if (gs.board(i)(j) == FLAG) 1.0 else 0.0
    }
    output
  }

  def gameStateToTrainingData(gs: GameState): (Array[Double], Array[Double]) = {
    val input = gameStateToNNInput(gs)
    val output = gameStateToNNOutput(gs)
    (input, output)
  }

  def toPlayer(player: QNNPlayer): Player[QNNPlayer] = new Player[QNNPlayer] {
    def bestMove(gs: GameState): Option[Move] = player.bestMove(gs)
  }

  def loadQNNPlayer(boardSize: Int, filename: String): QNNPlayer = {
    val restored: MultiLayerNetwork = ModelSerializer.restoreMultiLayerNetwork(filename)
    QNNPlayer(boardSize, restored)
  }
}
