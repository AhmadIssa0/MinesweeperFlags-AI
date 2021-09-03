
package xikai.minesweeper

/**  Typeclass Player.
  *  
  */
trait Player[A] {
  def bestMove(gamestate: GameState): Option[Move]
}

object Player {
  val randomMovePlayer = new Player[RandomMovePlayer] {
    def bestMove(gs: GameState) = RandomMovePlayer.bestMove(gs)
  }

  def playSingleGame[A, B](boardSize: Int,
    first: Player[A],
    second: Player[B]
  ): Vector[GameState] = {
    // Returns whether first player won
    def play(currGS: GameState, isFirstToMove: Boolean, history: Vector[GameState]): Vector[GameState] = {
      if (currGS.isGameOver) history
      else {
        val player = if (currGS.isBluesTurn) first else second
        val move = player.bestMove(currGS) match {
          case Some(m) => m
          case _ => throw new Exception(s"is random player: $isFirstToMove")
        }

        val nextGS: GameState = currGS.makeMove(move._1, move._2) match {
          case Right(gs) => gs
          case Left(error) =>
            throw new Exception(s"AI tries to play move: $move. But received error: $error")
        }
        play(nextGS, !isFirstToMove, history :+ nextGS)
      }
    }
    val init = GameState.createNewGame(boardSize)
    play(init, true, Vector(init))
  }


  // Returns number of games first player won.
  def playSeries[A,B](boardSize: Int, first: Player[A], second: Player[B], numOfGames: Int): Int = {
    // Returns whether first player won
    def firstWins(currGS: GameState, isFirstToMove: Boolean): Boolean = {
      if (currGS.isGameOver) currGS.blueScore >= currGS.redScore
      else {
        val player = if (currGS.isBluesTurn) first else second
        val move = player.bestMove(currGS).get
        val nextGS: GameState = currGS.makeMove(move._1, move._2) match {
          case Right(gs) => gs
          case Left(error) =>
            throw new Exception(s"AI tries to play move: $move. But received error: $error")
        }
        firstWins(nextGS, !isFirstToMove)
      }
    }

    (0 until numOfGames)
      .map(_ => firstWins(GameState.createNewGame(boardSize), true))
      .filter(_ == true)
      .length
  }

  // Returns avg number of flags first player won.
  def playSeriesScore[A,B](boardSize: Int, first: Player[A], second: Player[B], numOfGames: Int): Double = {
    // Sums up how many flags first player wins by

    // Returns whether first player won
    def firstWins(currGS: GameState, isFirstToMove: Boolean): Int = {
      if (currGS.isGameOver) currGS.blueScore - currGS.redScore
      else {
        val player = if (currGS.isBluesTurn) first else second
        val move = player.bestMove(currGS).get
        val nextGS: GameState = currGS.makeMove(move._1, move._2) match {
          case Right(gs) => gs
          case Left(error) =>
            throw new Exception(s"AI tries to play move: $move. But received error: $error")
        }
        firstWins(nextGS, !isFirstToMove)
      }
    }

    val score = 
    (0 until numOfGames)
      .map(_ => firstWins(GameState.createNewGame(boardSize), true))
      .sum
    score / (1.0 * numOfGames)
  }
}

