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
 *    FasterForest2.java
 *    Copyright (C) 2009 Fran Supek
 */

package cz.siret.prank.fforest2;

import cz.siret.prank.ffutils.ATimer;
import jdk.nashorn.internal.runtime.regexp.joni.exception.ValueException;
import weka.core.Instances;

import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;

/**
 * Stores a dataset that in FastRandomTrees use for training. The data points
 * are stored in a single-precision array indexed by attribute first, and then
 * by instance, to make access by FastRandomTrees faster. 
 * 
 * Also stores the sorted order of the instances by any attribute, can create
 * bootstrap samples, and seed a random number generator from the stored data.
 * 
 * @author Fran Supek (fran.supek[AT]irb.hr)
 * @author Jordi Pique (2.0 version)
 */
public class DataCache2 {

  /** Array with the indices of the selected attributes of the instances */
  protected int[] selectedAttributes;

  /** Indices of the attributes that are in sortedIndices. Categorical attributes can be selected for
   * a tree but they probably not be represented in sortedIndices */
  protected int[] attInSortedIndices;

  /** The dataset, first indexed by attribute, then by instance. */
  public final float[][] vals;

  /**
   * Attribute description - holds a 0 for numeric attributes, and the number
   * of available categories for nominal attributes.
   */
  protected final int[] attNumVals;

  /** The number of instances that will be selected for the inBag (an instance can be selected more than once */
  protected int bagSize;

  /** Numeric index of the class attribute. */
  protected final int classIndex;

  /** Number of attributes, including the class attribute. */
  protected int numAttributes;

  /** Number of classes. */
  protected final int numClasses;

  /** Number of instances. */
  protected final int numInstances;
  
  /** The class an instance belongs to. */
  protected final int[] instClassValues;

  /** Ordering of instances, indexed by attribute, then by instance. */ 
  protected int[][] sortedIndices;
  
  /** Weights of instances. */
  protected float[] instWeights;
  
  /** Is instance in 'bag' created by bootstrap sampling. */
  protected boolean[] inBag = null;
  /** How many instances are in 'bag' created by bootstrap sampling. */
  protected int numInBag = 0;

  /** Used in training of FastRandomTrees. */
  protected int[] whatGoesWhere = null;

  /** Array that will be used for a tree to store the indices of the instances that have a missing values for
   * a gives attribute */
  // protected int[] instancesMissVal;

  protected boolean isClassNominal;
  
  /**
   * Used in training of FastRandomTrees. Each tree can store its own
   * custom-seeded random generator in this field.
   */
  protected Random reusableRandomGenerator = null;


  /** Randomizes one attribute in the vals[][]; returns a copy of the vals[] 
   * before randomization. */
  public float[] scrambleOneAttribute( int attIndex, Random random ) {
    float[] toReturn = Arrays.copyOf( vals[attIndex], vals[attIndex].length );
    for ( int i=0; i < vals[attIndex].length; i++ ) {
      int swapWith = random.nextInt(vals[attIndex].length);
      float temp = vals[attIndex][i];
      vals[attIndex][i] = vals[attIndex][swapWith];
      vals[attIndex][swapWith] = temp;
    }
    return toReturn;
  }


  /**
   * Creates a DataCache by copying data from a weka.core.Instances object.
   *
   * @param origData
   */
  public DataCache2(Instances origData, int parallelism) throws Exception {

    classIndex = origData.classIndex();
    numAttributes = origData.numAttributes();
    numClasses = origData.numClasses();
    numInstances = origData.numInstances();

    isClassNominal = origData.classAttribute().isNominal();

    attNumVals = new int[origData.numAttributes()];
    for (int i = 0; i < attNumVals.length; i++) {
      if (origData.attribute(i).isNumeric()) {
        attNumVals[i] = 0;
      } else if (origData.attribute(i).isNominal()) {
        attNumVals[i] = origData.attribute(i).numValues();
      } else
        throw new Exception("Only numeric and nominal attributes are supported.");
    }

    /* Array is indexed by attribute first, to speed access in RF splitting. */
    vals = new float[numAttributes][numInstances];
    for (int a = 0; a < numAttributes; a++) {
      float[] valsA = vals[a];
      for (int i = 0; i < numInstances; i++) {

        valsA[i] = (float) origData.instance(i).value(a);  // deep copy
      }
    }

    instWeights = new float[numInstances];
    instClassValues = new int[numInstances];
    for (int i = 0; i < numInstances; i++) {
      instWeights[i] = (float) origData.instance(i).weight();
      instClassValues[i] = (int) origData.instance(i).classValue();
    }

    /* compute the sortedInstances for the whole dataset */
    
    sortedIndices = new int[numAttributes][];

    ForkJoinPool pool = new ForkJoinPool(parallelism);

    ATimer timer = ATimer.startTimer();
    for (int a = 0; a < numAttributes; a++) {
      if (a != classIndex) {
        sortedIndices[a] = cz.siret.prank.fforest.FastRfUtils.sortIndicesParallel(vals[a], parallelism, pool);
      }
    }
    System.out.println("time spent sorting: " + timer.getFormatted());

  }

  
  
  /**
   * Makes a copy of a DataCache. Most array fields are shallow copied, with the
   * exception of in inBag and whatGoesWhere arrays, which are created anew.
   * 
   * @param origData
   */
  public DataCache2(DataCache2 origData) {

    classIndex = origData.classIndex;       // copied
    numAttributes = origData.numAttributes; // copied
    numClasses = origData.numClasses;       // copied
    numInstances = origData.numInstances;   // copied

    attNumVals = origData.attNumVals;       // shallow copied
    instClassValues =
            origData.instClassValues;       // shallow copied
    vals = origData.vals;                   // shallow copied - very big array!
    sortedIndices = origData.sortedIndices; // shallow copied - also big

    instWeights = origData.instWeights;     // shallow copied

    inBag = new boolean[numInstances];      // gets its own inBag array

    numInBag = 0;
    
    whatGoesWhere = null;     // this will be created when tree building starts

    isClassNominal = origData.isClassNominal;
  }

  
  
  /**
   * Uses sampling with replacement to create a new DataCache from an existing
   * one.
   * 
   * The probability of sampling a specific instance does not depend on its
   * weight. When an instance is sampled multiple times, its weight in the new
   * DataCache increases to a multiple of the original weight.
   * 
   * a bootstrap sample (n of of in
   * @param random A random number generator.
   * @return a new DataCache - consult "DataCache(DataCache origData)"
   * constructor to see what's deep / shallow copied
   */
  public DataCache2 resample(Random random, int nAttrVirtual) {
    if (nAttrVirtual >= numAttributes) {
      throw new ValueException("nAttr must be less than numAttributes");
    }
    // makes shallow copy of vals matrix
    // makes a deep copy of each instance, but with a shallow copy of its attributes
    // TODO create DataCacheView to avion allocating this and int[] inBag in the constructor
    DataCache2 result = new DataCache2(this);

    result.reusableRandomGenerator = random;
    // Time ~ 160908 ns
    float[] newWeights = new float[ numInstances ]; // all 0.0 by default
    
    for ( int r = 0; r < bagSize; r++ ) {
      
      int curIdx = random.nextInt( numInstances );
      newWeights[curIdx] += instWeights[curIdx];
      if ( !result.inBag[curIdx] ) {
        result.numInBag++;
        result.inBag[curIdx] = true;
      }
    }
    result.instWeights = newWeights;

    // select the subset of features
    result.selectedAttributes = new int[nAttrVirtual];
    int[] permIndices = FastRfUtils.randomPermutation(numAttributes, random);
    int nAttInSortedIndices = 0;
    for (int i = 0; i < nAttrVirtual; ++i) {
      int a = permIndices[i];
      if (a == classIndex)
        // swap the classIndex with the first element not included
        a = permIndices[nAttrVirtual];

      result.selectedAttributes[i] = a; // it will never have the attribute class
      nAttInSortedIndices += isAttrNominal(a) ? 0 : 1;
    }
    result.attInSortedIndices = new int[nAttInSortedIndices];
    result.whatGoesWhere = new int[ result.inBag.length ];

    // Time random access to the weights of all the instances:
    //    - For newWeights[] ~ 18540 ns
    //    - For instances.get(_).weight() ~ 72177 ns

    // we also need to fill sortedIndices by peeking into the inBag array
    // we will use the "createInBagSortedIndices()" for this

    return result;

  }

  /** Invoked only when tree is trained. */
  protected void createInBagSortedIndicesNew() {

    int[][] newSortedIndices = new int[ numAttributes ][ ];
    // instancesMissVal = new int[numInBag];
    int idx = 0;
    int maxLvl = 0; // maximum number of values for the categorical features

    boolean allCategorical = attInSortedIndices.length == 0;
    if (allCategorical) attInSortedIndices = new int[1];

    for (int a : selectedAttributes) {
      // we will add, at most, only one categorical feature in sortedIndices

      attInSortedIndices[idx] = a;
      ++idx;

      int[] sortedIndicesA = sortedIndices[a];
      int[] newSortedIndicesA = new int[this.numInBag];
      newSortedIndices[a] = newSortedIndicesA;

      int inBagIdx = 0;
      for (int origIdx : sortedIndicesA) {
        if (inBag[origIdx]) {
          newSortedIndicesA[inBagIdx] = origIdx;
          inBagIdx++;
        }
      }
    }
    this.sortedIndices = newSortedIndices;
  }


  
  /** Does the given attribute - instance combination contain a missing value? */
  public final boolean isValueMissing( int attIndex, int instIndex ) {
    return this.vals[attIndex][instIndex] == Float.MAX_VALUE;
  }

  
  /** Is an attribute with the given index nominal? */
  public final boolean isAttrNominal( int attIndex ) {
    return attNumVals[attIndex] > 0;
  }
  

  /**
   * Returns a random number generator. The initial seed of the random
   * number generator depends on the given seed and the contents of the
   * sortedIndices array (a single attribute is picked, a hashcode of its
   * sortedIndices is computed).
   *
   * Implementation from https://github.com/fracpete/fastrandomforest-weka-package/blob/master/src/main/java/hr/irb/fastRandomForest/DataCache.java
   *
   * @param seed the given seed
   * @return the random number generator
   */
  public Random getRandomNumberGenerator(long seed) {

    Random r = new Random(seed);

    // ignore data signature since sortedIndices are not sorted in a stable way
    //long dataSignature = Arrays.hashCode( sortedIndices[ r.nextInt( numAttributes ) ] );
    //r.setSeed( dataSignature + seed );

    return r;

  }
}