
package xikai.minesweeper

import scala.util.Random
import scala.collection.immutable.Queue
/** 
  * @param board The revealed board. (See package.scala for Board type).
  * @param visible Which squares are visible to both players (?).
  * @param blueFlagsCaptured Coordinates (x, y) with 0 <= x,y < gridSize of flags captured.
  * 
  */
case class GameState(
  board: Board,
  visible: Vector[Vector[Boolean]],
  isBluesTurn: Boolean,
  blueFlagsCaptured: Vector[(Int, Int)],
  redFlagsCaptured: Vector[(Int, Int)],
  totalFlags: Int
  ) {
  import GameState._
  lazy val blueScore = blueFlagsCaptured.length
  lazy val redScore = redFlagsCaptured.length
  lazy val boardSize = board.length
  lazy val winningScore = totalFlags / 2 + 1

  def isGameOver: Boolean = 
    blueScore >= winningScore || redScore >= winningScore

  override def toString: String = {
    val boardStr =
      for (j <- 0 until boardSize) yield {
        for (i <- 0 until boardSize) yield {
          if (!visible(i)(j)) "-"
          else if (board(i)(j) == FLAG) "*"
          else board(i)(j).toString
        }
      }
    boardStr.map(row => row.mkString(" ")).mkString("\n")
  }

  def reverseRedAndBlue: GameState = {
    GameState(board, visible, !isBluesTurn, redFlagsCaptured, blueFlagsCaptured, totalFlags)
  }


  /**
    * Returns either an error message or a new GameState if
    * move is valid.
    */
  def makeMove(x: Int, y: Int): Either[String, GameState] = {
    assert (0 <= x && x < boardSize && 0 <= y && y < boardSize,
      s"($x,$y) is not coordinate of $boardSize x $boardSize board")

    if (visible(x)(y)) Left(s"Game square ($x,$y) is already revealed")
    else if (isGameOver) Left(s"Game is over.")
    else if (board(x)(y) == FLAG) {
      val newVisible = createVisible({
        case (`x`,`y`) => true // Square player clicked on is now visible
        case (x, y) => visible(x)(y)
      }, boardSize)
      val newBlueFlagsCaptured =
        if (isBluesTurn) (blueFlagsCaptured :+ (x,y)) else blueFlagsCaptured
      val newRedFlagsCaptured =
        if ( ! isBluesTurn) (redFlagsCaptured :+ (x,y)) else redFlagsCaptured
      Right(GameState(
        board = board,
        visible = newVisible,
        isBluesTurn = isBluesTurn,
        blueFlagsCaptured = newBlueFlagsCaptured,
        redFlagsCaptured = newRedFlagsCaptured,
        totalFlags = totalFlags
      ))
    } else if (board(x)(y) == 0) {
      val connect = connectedGrid(Queue((x,y)), Vector((x,y)))

      val newVisible = createVisible({
        //case (`x`,`y`) => true // Square player clicked on is now visible
        case (x, y) => (connect contains (x, y)) || visible(x)(y)
      }, boardSize)
      Right(GameState(
        board = board,
        visible = newVisible,
        isBluesTurn = ! isBluesTurn,
        blueFlagsCaptured = blueFlagsCaptured,
        redFlagsCaptured = redFlagsCaptured,
        totalFlags = totalFlags
      ))
    } else {
      val newVisible = createVisible({
        case (`x`,`y`) => true // Square player clicked on is now visible
        case (x, y) => visible(x)(y)
      }, boardSize)
      Right(GameState(
        board = board,
        visible = newVisible,
        isBluesTurn = ! isBluesTurn,
        blueFlagsCaptured = blueFlagsCaptured,
        redFlagsCaptured = redFlagsCaptured,
        totalFlags = totalFlags
      ))
    }
  }

  def connectedGrid(q: Queue[(Int, Int)], list: Vector[(Int, Int)]): Vector[(Int, Int)] = {
    if (q.isEmpty) list
    else {
      val (s, qs) = q.dequeue
      val (x0, y0) = s

      val neighbours =
        for (x <- (x0 - 1 to x0 + 1).toVector;
             y <- (y0 - 1 to y0 + 1).toVector;
             if ((x,y) != (x0,y0) && 0 <= x && x < boardSize && 0 <= y && y < boardSize))
        yield (x, y)

      val nbrsInComponent = neighbours.filter { case (x, y) =>
        board(x)(y) != FLAG && ! list.contains((x, y))
      }

      val nbrsToEnqueue = nbrsInComponent.filter { case (x, y) => board(x)(y) == 0 }

      connectedGrid(qs ++ nbrsToEnqueue, list ++ nbrsInComponent)
    }
  }

}

object GameState {
  def createNewGameWithRNG(random: Random, boardSize: Int): GameState = {
    val totalFlags = {
      val m = boardSize / 2
      if (boardSize == 5) 3
      else if (m % 2 == 1) m*m else (m-1)*(m-1)
    }

    val flags = Stream.continually((random.nextInt(boardSize), random.nextInt(boardSize)))
      .distinct
      .take(totalFlags)
      .toVector

    val board = createBoard({ case (x,y) =>
        if (flags.contains((x,y))) FLAG
        else { // Not a flag, find out how many neighbours are flags
          val neighbours = Vector(
            (x-1,y-1), (x-1,y), (x-1,y+1), (x,y-1),
            (x,y+1), (x+1,y-1), (x+1,y), (x+1,y+1))
          neighbours.count(coord => flags.contains(coord))
        }
      }, boardSize)

    val visible = Vector.fill(boardSize, boardSize)(false)

    GameState(
      board = board,
      visible = visible,
      isBluesTurn = true, // Blue starts the game
      blueFlagsCaptured = Vector(),
      redFlagsCaptured = Vector(),
      totalFlags = totalFlags
    )
  }

  /**  Create a new game given by a random seed.
    */
  def createNewGameWithSeed(seed: Int, boardSize: Int): GameState = 
    createNewGameWithRNG(new Random(seed), boardSize)

  def createNewGame(boardSize: Int): GameState = 
    createNewGameWithRNG(new Random(), boardSize)

  def createBoard(f: (Int, Int) => SquareState, boardSize: Int): Board = {
    Vector.tabulate[SquareState](boardSize, boardSize)(f)
  }

  def createVisible(f: (Int, Int) => Boolean, boardSize: Int) = {
    Vector.tabulate[Boolean](boardSize, boardSize)(f)
  }

}
