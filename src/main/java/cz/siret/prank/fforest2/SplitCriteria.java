/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin Street, Suite 500, Boston, MA 02110.
 */

/*
 *    SplitCriteria.java
 *    Copyright (C) 1999 University of Waikato, Hamilton, NZ (original
 *      code, ContingencyTables.java )
 *    Copyright (C) 2008 Fran Supek (adapted code)
 */

package cz.siret.prank.fforest2;


/**
 * Functions used for finding best splits in FastRfTree. Based on parts of
 * weka.core.ContingencyTables, revision 1.7, by Eibe Frank
 * 
 * @author Eibe Frank (eibe@cs.waikato.ac.nz) - original code
 * @author Fran Supek (fran.supek[AT]irb.hr) - adapted code
 * @author Jordi Pique (2.0 version)
 * @version 2.0
 */
public class SplitCriteria {
  
  
  /**
   * Similar to weka.core.ContingencyTables.entropyConditionedOnRows.
   * 
   * Does not output entropy, output is modified to make routine faster:
   * the result is not divided by "total", as the total is a constant
   * in all operations (subtraction, comparison) performed as a part of
   * splitting in FastRfTree. Also, we don't have to divide by Math.log(2)
   * as the logarithms provided by fastLog2() are already base 2.
   * 
   * @param matrix the contingency table
   * @return the conditional entropy of the columns given the rows
   */
  public static double entropyConditionedOnRows(float[][] matrix) {

    float returnValue = 0, sumForBranch;
    //double total = 0;

    for (float[] branch : matrix) {
      sumForBranch = 0;
      for (float v : branch) {
        returnValue = returnValue + lnFunc(v);
        sumForBranch += v;
      }
      returnValue = returnValue - lnFunc(sumForBranch);
      // total += sumForRow;
    }

    //return -returnValue / (total * log2);
    return -returnValue; 
         
  }

  public static float giniConditionedOnRows(float[][] matrix) {

    float returnValue = 0;
    float auxSum;
    float sumForBranch;

    for (float[] branch : matrix) {
      auxSum = 0;
      sumForBranch = 0;

      for (float v : branch) {
        auxSum += v * v;
        sumForBranch += v;
      }
      returnValue += sumForBranch - auxSum / sumForBranch;
    }

    return returnValue;
  }

  public static float giniConditionedOnRowsLR(float[] distL, float[] distR) {
    return giniRow(distL) + giniRow(distR);
  }

  public static float giniRow(float[] dist) {
    float auxSum = 0;
    float sumForBranch = 0;

    for (float v : dist) {
      auxSum += v * v;
      sumForBranch += v;
    }

    return sumForBranch - auxSum / sumForBranch;
  }

  public static float giniConditionedOnRowsLR2(float distL0, float distL1, float distR0, float distR1) {
    float auxSum = distL0*distL0 + distL1*distL1;
    float sumForBranch = distL0 + distL1;
    float res = sumForBranch - auxSum / sumForBranch;

    auxSum = distR0*distR0 + distR1*distR1;
    sumForBranch = distR0 + distR1;
    res += sumForBranch - auxSum / sumForBranch;

    return res;
  }


//  public static double giniConditionedOnRows(double[][] matrix) {
//
//    double returnValue = 0, sumForBranch, auxSum;
//
//    for (int branchNum = 0; branchNum < matrix.length; branchNum++) {
//      auxSum = matrix[branchNum][0] * matrix[branchNum][0];
//      sumForBranch = matrix[branchNum][0];
//      for (int classNum = 1; classNum < matrix[0].length; classNum++) {
//        auxSum += matrix[branchNum][classNum] * matrix[branchNum][classNum];
//        sumForBranch += matrix[branchNum][classNum];
//      }
//      returnValue += sumForBranch - auxSum/sumForBranch;
//    }
//
//    return returnValue;
//  }

  /**
   * Similar to weka.core.ContingencyTables.entropyOverColumns
   * 
   * Does not output entropy, output is modified to make routine faster:
   * the result is not divided by "total", as the total is a constant
   * in all operations (subtraction, comparison) performed as a part of
   * splitting in FastRfTree. Also, we don't have to divide by Math.log(2)
   * as the logarithms provided by fastLog2() are already base 2.
   *   
   * @param matrix the contingency table
   * @return the columns' entropy
   */
  public static double entropyOverColumns(float[][] matrix) {

    //return ContingencyTables.entropyOverColumns(matrix);

    float returnValue = 0, sumForColumn, total = 0;

    for (int j = 0; j < matrix[0].length; j++) {
      sumForColumn = 0;
      for (float[] floats : matrix) {
        sumForColumn += floats[j];
      }
      returnValue -= lnFunc(sumForColumn);
      total += sumForColumn;
    }

    //return (returnValue + lnFunc(total)) / (total * log2);
    return (returnValue + lnFunc(total));

  }

  public static float giniOverColumns(float[][] matrix) {

    float auxSum = 0, sumForColumn, total = 0;

    for (int j = 0; j < matrix[0].length; j++) {
      sumForColumn = matrix[0][j];
      for (int i = 1; i < matrix.length; i++) {
        sumForColumn += matrix[i][j];
      }
      auxSum += sumForColumn * sumForColumn;
      total += sumForColumn;
    }

    return total - auxSum/total;
  }

  
  
  /**
   * A fast approximation of log base 2, in single precision. Approximately
   * 4 times faster than Java's Math.log() function.
   * 
   * Inspired by C code by Laurent de Soras:
   * http://www.flipcode.com/archives/Fast_log_Function.shtml
   */
   public static float fastLog2( float val ) {
     
     int bits = Float.floatToIntBits(val);
     
     final int log_2 = ( (bits >> 23) & 255) - 128;
     bits &= ~(255 << 23);
     bits += 127 << 23;
     
     val = Float.intBitsToFloat(bits);
     
     val = ((-1.0f/3) * val + 2) * val - 2.0f/3;
     return (val + log_2);

   }
  
  
  
  /**
   * Help method for computing entropy.
   */
  public static float lnFunc(float num) {

    if (num <= 1e-6) {
      return 0;
    } else {
      return num * fastLog2( num );
      //return num * Math.log( num );
    }
    
  }

  

  
  
}
