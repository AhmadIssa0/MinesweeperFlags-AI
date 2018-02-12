
package xikai

/**  The package object is where we put functions, enumerations, type definitions etc
  *  which we want to be available throughout our package, and which don't belong to a class.
  */

package object minesweeper {
  type Board = Vector[Vector[Int]]
  type SquareState = Int
  type Move = (Int, Int)
  val FLAG = -1
}
