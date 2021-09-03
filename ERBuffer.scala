

package xikai.minesweeper

class ERBuffer[A](size: Int) {
 /*  Experience Replay buffer for deep q-learning.
  * 
  */

  var buffer: Vector[A] = Vector()

  def isFilled = buffer.length == size

  def add(a: A): ERBuffer[A] = {
    if (isFilled) {
      buffer = buffer.drop(1)
    }
    buffer = buffer :+ a
    this
  }

  def batch(n: Int): Vector[A] = {
    import scala.util.Random;
    Random.shuffle(buffer).take(n).toVector
  }
}
