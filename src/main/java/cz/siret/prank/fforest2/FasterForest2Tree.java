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
 *    FasterForest2Tree.java
 *    Copyright (C) 2001 University of Waikato, Hamilton, NZ (original code,
 *      RandomTree.java)
 *    Copyright (C) 2013 Fran Supek (adapted code)
 */

package cz.siret.prank.fforest2;

import cz.siret.prank.fforest.FasterTree;
import weka.classifiers.AbstractClassifier;
import weka.core.*;
import weka.core.Capabilities.Capability;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;


/**
 * Based on the "weka.classifiers.trees.RandomTree" class, revision 1.19,
 * by Eibe Frank and Richard Kirkby, with major modifications made to improve
 * the speed of classifier training.
 * 
 * Please refer to the Javadoc of buildTree, splitData and distribution
 * function, as well as the changelog.txt, for the details of changes to 
 * FasterForest2Tree.
 * 
 * This class should be used only from within the FasterForest2 classifier.
 * 
 * @author Eibe Frank (eibe@cs.waikato.ac.nz) - original code
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz) - original code
 * @author Fran Supek (fran.supek[AT]irb.hr) - adapted code
 * @author Jordi Pique (2.0 version)
 * @version $Revision: 2.0$
 */
class FasterForest2Tree
        extends AbstractClassifier
        implements OptionHandler, WeightedInstancesHandler, Runnable {

  /** for serialization */
  static final long serialVersionUID = 8934314652175299375L;

  /** A reference to the data.inBag field, in order to know which are the inBag instances for this tree after
   * it has been built and data has been destroyed */
  public boolean[] myInBag;

  /** The seed for the Random Generator for this tree */
  public int m_seed;

  /** A HashSet with the attributes used to build this tree */
  public HashSet<Integer> subsetSelectedAttr;
  
  /** The subtrees appended to this tree (node). */
  protected FasterForest2Tree[] m_Successors;

  /** For access to parameters of the RF (k, or maxDepth). */
  protected FasterForest2 m_MotherForest;

  /** The attribute to split on. */
  protected int m_Attribute = -1;

  /** The split point. */
  protected float m_SplitPoint = Float.NaN;
  
  /** The proportions of training instances going down each branch. */
  protected float[] m_Prop = null;

  /** Class probabilities from the training vals. */
  protected float[] m_ClassProbs = null;

  /** The dataset used for training. */
  protected transient DataCache data = null;
  
  /**
   * Since 0.99: holds references to temporary arrays re-used by all nodes
   * in the tree, used while calculating the "props" for various attributes in
   * distributionSequentialAtt(). This is meant to avoid frequent 
   * creating/destroying of these arrays.
   */
  //protected transient float[] tempProps;
  
  /**
   * Since 0.99: holds references to temporary arrays re-used by all nodes
   * in the tree, used while calculating the "dists" for various attributes
   * in distributionSequentialAtt(). This is meant to avoid frequent 
   * creating/destroying of these arrays.
   */
  //protected transient double[][] tempDists;
  //protected transient double[][] tempDistsOther;

  //protected transient float[] tempDistsL;
  //protected transient float[] tempDistsR;
  //protected transient float[] tempDistsOtherL;
  //protected transient float[] tempDistsOtherR;

  /** Minimum number of instances for leaf. */
  protected static final int m_MinNum = 1;

  protected static final int m_MinInstancesForSplit = Math.max(2, m_MinNum);

  /**
   * This constructor should not be used. Instead, use the next two constructors
   */
  @Deprecated
  public FasterForest2Tree() {
  }

  /**
   * Constructor for the first node of the tree
   * @param motherForest
   * @param data
   */
  public FasterForest2Tree(FasterForest2 motherForest, DataCache data, int seed) {
    //int numClasses = data.numClasses;
    this.m_seed = seed;
    this.data = data;
    // all parameters for training will be looked up in the motherForest (maxDepth, k_Value)
    this.m_MotherForest = motherForest;
    // 0.99: reference to these arrays will get passed down all nodes so the array can be re-used 
    // 0.99: this array is of size two as now all splits are binary - even categorical ones
    //this.tempProps = new float[2];

    //this.tempDistsL = new float[numClasses];
    //this.tempDistsR = new float[numClasses];
    //this.tempDistsOtherL = new float[numClasses];
    //this.tempDistsOtherR = new float[numClasses];

    //this.tempDists = new double[2][];
    //this.tempDists[0] = new double[numClasses];
    //this.tempDists[1] = new double[numClasses];
    //this.tempDistsOther = new double[2][];
    //this.tempDistsOther[0] = new double[numClasses];
    //this.tempDistsOther[1] = new double[numClasses];
  }

  /**
   * Constructor for all the nodes except the root
   * @param motherForest
   * @param data
   */
  public FasterForest2Tree(FasterForest2 motherForest, DataCache data /*, float[] tempDistsL, float[] tempDistsR,
                           float[] tempDistsOtherL, float[] tempDistsOtherR, float[] tempProps*/) {
    this.m_MotherForest = motherForest;
    this.data = data;
    // new in 0.99 - used in distributionSequentialAtt()
    //this.tempDists = tempDists;
    //this.tempDistsOther = tempDistsOther;

    //this.tempDistsL = tempDistsL;
    //this.tempDistsR = tempDistsR;
    //this.tempDistsOtherL = tempDistsOtherL;
    //this.tempDistsOtherR = tempDistsOtherR;

    //this.tempProps = tempProps;
  }

  /**
   * Get the value of MinNum.
   *
   * @return Value of MinNum.
   */
  public final int getMinNum() {

    return m_MinNum;
  }


  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String KValueTipText() {
    return "Sets the number of randomly chosen attributes.";
  }


  /**
   * Get the value of K.
   * @return Value of K.
   */
  public final int getKValue() {
    return m_MotherForest.m_KValue;
  }


  /**
   * Get the maximum depth of the tree, 0 for unlimited.
   * @return 		the maximum depth.
   */
  public final int getMaxDepth() {
    return m_MotherForest.m_MaxDepth;
  }


  /**
   * Returns default capabilities of the classifier.
   * @return      the capabilities of this classifier
   */
  @Override
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.disableAll(); 

    // attributes
    //result.enable(Capability.NOMINAL_ATTRIBUTES);
    result.enable(Capability.NUMERIC_ATTRIBUTES);
    //result.enable(Capability.DATE_ATTRIBUTES);
    //result.enable(Capability.MISSING_VALUES);

    // class
    result.enable(Capability.NOMINAL_CLASS);
    //result.enable(Capability.MISSING_CLASS_VALUES);

    return result;
  }


  /**
   * This function is not supported by FasterForest2Tree, as it requires a
   * DataCache for training.

   * @throws Exception every time this function is called
   */
  @Override
  public void buildClassifier(Instances data) throws Exception {
    throw new Exception("FasterForest2Tree can be used only by FasterForest2 " +
            "and FastRfBagger classes, not directly.");
  }



  /**
   * Builds classifier. Makes the initial call to the recursive buildTree 
   * function. The name "run()" is used to support multithreading via an
   * ExecutorService.
   */
  public void run() {
    // makes a copy of data and selects randomly which are the inBag instances and the subset of features
    data = data.resample(data.getRandomNumberGenerator(m_seed), m_MotherForest.m_numFeatTree);
    // we need to save the inBag[] array in order to have access to it after this.data is destroyed
    myInBag = data.inBag;

    // compute initial class counts
    float[] classProbs = new float[data.numClasses];
    for (int i = 0; i < data.numInstances; i++) {
      classProbs[data.instClassValues[i]] += data.instWeights[i];
    }

    // Creates a HashSet in order to know which attributes are used in this tree
    subsetSelectedAttr = new HashSet<>(data.selectedAttributes.length);
    for (int attr : data.selectedAttributes) subsetSelectedAttr.add(attr);

    // create the attribute indices window - skip class
    int[] attIndicesWindow = data.selectedAttributes;
    // create the sorted indices matrix
    data.createInBagSortedIndicesNew();
    // first recursive call
    buildTree(data.sortedIndices, 0, data.numInBag - 1,
            classProbs, attIndicesWindow, 0);

    this.data = null;
//    int nNodes = countNodes();
//    Benchmark.updateNumNodes(nNodes);
  }

  

  /**
   * Computes class distribution of an instance using the FasterForest2Tree.<p>
   *
   * In Weka's RandomTree, the distributions were normalized so that all
   * probabilities sum to 1; this would abolish the effect of instance weights
   * on voting. In FasterForest2 0.97 onwards, the distributions are
   * normalized by dividing with the number of instances going into a leaf.<p>
   * 
   * @param instance the instance to compute the distribution for
   * @return the computed class distribution
   * @throws Exception if computation fails
   */
  @Override
  public double[] distributionForInstance(Instance instance) throws Exception {

    double[] returnedDist = null;

    if (m_Attribute != -1) {  // ============================ node is not a leaf

//      if (instance.isMissing(m_Attribute)) {  // ---------------- missing value
//
//        returnedDist = new double[m_MotherForest.m_Info.numClasses()];
//        // split instance up
//        for (int i = 0; i < m_Successors.length; i++) {
//          double[] help = m_Successors[i].distributionForInstance(instance);
//          if (help != null) {
//            for (int j = 0; j < help.length; j++) {
//              returnedDist[j] += m_Prop[i] * help[j];
//            }
//          }
//        }
//
//      } else if (m_MotherForest.m_Info
//              .attribute(m_Attribute).isNominal()) { // ------ nominal
//
//        //returnedDist = m_Successors[(int) instance.value(m_Attribute)]
//        //        .distributionForInstance(instance);
//
//        // 0.99: new - binary splits (also) for nominal attributes
//        if ( instance.value(m_Attribute) == m_SplitPoint ) {
//          returnedDist = m_Successors[0].distributionForInstance(instance);
//        } else {
//          returnedDist = m_Successors[1].distributionForInstance(instance);
//        }
//
//
//      } else { // ------------------------------------------ numeric attributes

        if (instance.value(m_Attribute) < m_SplitPoint) {
          returnedDist = m_Successors[0].distributionForInstance(instance);
        } else {
          returnedDist = m_Successors[1].distributionForInstance(instance);
        }
//      }

      return returnedDist;

    } else { // =============================================== node is a leaf

      return FastRfUtils.toDoubles2(m_ClassProbs);

    }

  }


//  /**
//   * Computes class distribution of an instance using the FasterForest2Tree. <p>
//   *
//   * Works correctly only if the DataCache has the same attributes as the one
//   * used to train the FasterForest2Tree - but this function does not check for
//   * that! <p>
//   *
//   * Main use of this is to compute out-of-bag error (also when finding feature
//   * importances).
//   *
//   * @param instIdx the index of the instance to compute the distribution for
//   * @return the computed class distribution
//   * @throws Exception if computation fails
//   */
//  public double[] distributionForInstanceInDataCache(DataCache data, int instIdx) {
//    FasterForest2Tree frt = this;
//    while (frt.m_Attribute > -1) {
//      // 0.99: new - binary splits (also) for nominal attributes
//      if (data.isAttrNominal(frt.m_Attribute)) {
//        if ( data.vals[frt.m_Attribute][instIdx] == frt.m_SplitPoint ) {
//          frt = (FasterForest2Tree) frt.m_Successors[0];
//        } else {
//          frt = (FasterForest2Tree) frt.m_Successors[1];
//        }
//      } else {
//        if (data.vals[frt.m_Attribute][instIdx] < frt.m_SplitPoint) {
//          frt = (FasterForest2Tree) frt.m_Successors[0];
//        } else {
//          frt = (FasterForest2Tree) frt.m_Successors[1];
//        }
//      }
//    }
//    return frt.m_ClassProbs;
//  }

  /**
   * Computes class distribution of an instance using the FasterForest2Tree. <p>
   *
   * Works correctly only if the DataCache has the same attributes as the one
   * used to train the FasterForest2Tree - but this function does not check for
   * that! <p>
   *
   * Main use of this is to compute out-of-bag error (also when finding feature
   * importances).
   *
   * @param instIdx the index of the instance to compute the distribution for
   * @return the computed class distribution
   * @throws Exception if computation fails
   */
  public double[] distributionForInstanceInDataCache(DataCache data, int instIdx) {

    if (m_Attribute != -1) {  // ============================ node is not a leaf
      double[] returnedDist = null;

//      if ( data.isValueMissing(m_Attribute, instIdx) ) {  // ---------------- missing value
//
//        returnedDist = new double[m_MotherForest.m_Info.numClasses()];
//        // split instance up
//        for (int i = 0; i < m_Successors.length; i++) {
//          double[] help = m_Successors[i].distributionForInstanceInDataCache(data, instIdx);
//          if (help != null) {
//            for (int j = 0; j < help.length; j++) {
//              returnedDist[j] += m_Prop[i] * help[j];
//            }
//          }
//        }
//
//      } else if ( data.isAttrNominal(m_Attribute) ) { // ------ nominal
//        // 0.99: new - binary splits (also) for nominal attributes
//        if ( data.vals[m_Attribute][instIdx] == m_SplitPoint ) {
//          returnedDist = m_Successors[0].distributionForInstanceInDataCache(data, instIdx);
//        } else {
//          returnedDist = m_Successors[1].distributionForInstanceInDataCache(data, instIdx);
//        }
//
//      } else { // ------------------------------------------ numeric attributes
        if ( data.vals[m_Attribute][instIdx] < m_SplitPoint) {
          returnedDist = m_Successors[0].distributionForInstanceInDataCache(data, instIdx);
        } else {
          returnedDist = m_Successors[1].distributionForInstanceInDataCache(data, instIdx);
        }
//      }
      return returnedDist;
//
    } else { // =============================================== node is a leaf
      return FastRfUtils.toDoubles2(m_ClassProbs);
    }
  }

  private int countNodes() {
    if (m_Attribute != -1) {
      int result = 1;
      if (m_Successors[0] instanceof FasterForest2Tree)  {
        result += ((FasterForest2Tree) m_Successors[0]).countNodes();
      }
      if (m_Successors[1] instanceof FasterForest2Tree)  {
        result += ((FasterForest2Tree) m_Successors[1]).countNodes();
      }
      return result;
    } else {
      return 1;
    }
  }
  
 /**
   * Recursively generates a tree. A derivative of the buildTree function from
   * the "weka.classifiers.trees.RandomTree" class, with the following changes
   * made:
   * <ul>
   *
   * <li>m_ClassProbs are now remembered only in leaves, not in every node of
   *     the tree
   *
   * <li>m_Distribution has been removed
   *
   * <li>members of dists, splits, props and vals arrays which are not used are
   *     dereferenced prior to recursion to reduce memory requirements
   *
   * <li>a check for "branch with no training instances" is now (FastRF 0.98)
   *     made before recursion; with the current implementation of splitData(),
   *     empty branches can appear only with nominal attributes with more than
   *     two categories
   *
   * <li>each new 'tree' (i.e. node or leaf) is passed a reference to its
   *     'mother forest', necessary to look up parameters such as maxDepth and K
   *
   * <li>pre-split entropy is not recalculated unnecessarily
   *
   * <li>uses DataCache instead of weka.core.Instances, the reference to the
   *     DataCache is stored as a field in FasterForest2Tree class and not passed
   *     recursively down new buildTree() calls
   *
   * <li>similarly, a reference to the random number generator is stored
   *     in a field of the DataCache
   *
   * <li>m_ClassProbs are now normalized by dividing with number of instances
   *     in leaf, instead of forcing the sum of class probabilities to 1.0;
   *     this has a large effect when class/instance weights are set by user
   *
   * <li>a little imprecision is allowed in checking whether there was a
   *     decrease in entropy after splitting
   * 
   * <li>0.99: the temporary arrays splits, props, vals now are not wide
   * as the full number of attributes in the dataset (of which only "k" columns
   * of randomly chosen attributes get filled). Now, it's just a single array
   * which gets replaced as the k features are evaluated sequentially, but it
   * gets replaced only if a next feature is better than a previous one.
   * 
   * <li>0.99: the SortedIndices are now not cut up into smaller arrays on every
   * split, but rather re-sorted within the same array in the splitDataNew(),
   * and passed down to buildTree() as the original large matrix, but with
   * start and end points explicitly specified
   * 
   * </ul>
   * 
   * @param sortedIndices the indices of the instances of the whole bootstrap replicate
   * @param startAt First index of the instance to consider in this split; inclusive.
   * @param endAt Last index of the instance to consider; inclusive.
   * @param classProbs the class distribution
   * @param attIndicesWindow the attribute window to choose attributes from
   * @param depth the current depth
   */
  protected void buildTree(int[][] sortedIndices, int startAt, int endAt,
          float[] classProbs,
          int[] attIndicesWindow,
          int depth)  {

    int sortedIndicesLength = endAt - startAt + 1;

    // Check if node doesn't contain enough instances or is pure 
    // or maximum depth reached, make leaf.
    if ( ( sortedIndicesLength < m_MinInstancesForSplit )  // small
            || FastRfUtils.isPureDist(classProbs[0], classProbs[1])       // pure
            || ( depth >= m_MotherForest.m_MaxDepth && m_MotherForest.m_MaxDepth > 0 )                           // deep
            ) {
      m_Attribute = -1;  // indicates leaf (no useful attribute to split on)
      
      // normalize by dividing with the number of instances (as of ver. 0.97)
      // unless leaf is empty - this can happen with splits on nominal
      // attributes with more than two categories
      if ( sortedIndicesLength != 0 ) {
        //for (int c = 0; c < classProbs.length; c++) {
        //  classProbs[c] /= sortedIndicesLength;
        //}
        classProbs[0] /= sortedIndicesLength;
        classProbs[1] /= sortedIndicesLength;
      }
      m_ClassProbs = classProbs;
      this.data = null;
      return;
    } // (leaf making)
    
    // new 0.99: all the following are for the best attribute only! they're updated while sequentially through the attributes
    float val = Float.NaN; // value of splitting criterion
    float[][] dist = new float[2][data.numClasses];  // class distributions (contingency table), indexed first by branch, then by class
    float[] prop = new float[2]; // the branch sizes (as fraction)
    float split = Float.NaN;  // split point

    // Investigate K random attributes
    int attIndex = 0;
    int windowSize = attIndicesWindow.length;
    int k = getKValue();
    boolean sensibleSplitFound = false;
    float prior = Float.NaN;
    float bestNegPosterior = -Float.MAX_VALUE;
    int bestAttIdx = -1;

    Random random = data.reusableRandomGenerator;

    while ((windowSize > 0) && (k-- > 0 || !sensibleSplitFound ) ) {

      int chosenIndex = random.nextInt(windowSize);
      attIndex = attIndicesWindow[chosenIndex];

      // shift chosen attIndex out of window
      windowSize--;
      attIndicesWindow[chosenIndex] = attIndicesWindow[windowSize];
      attIndicesWindow[windowSize] = attIndex;

      // new: 0.99
//      long t = System.nanoTime();
      float candidateSplit = distributionSequentialAtt( prop, dist,
              bestNegPosterior, attIndex, 
              sortedIndices[attIndex], startAt, endAt, classProbs);
//      Benchmark.updateTime(System.nanoTime() - t);


      if ( Float.isNaN(candidateSplit) ) {
        continue;  // we did not improve over a previous attribute! "dist" is unchanged from before
      }
      // by this point we know we have an improvement, so we keep the new split point
      split = candidateSplit;
      bestAttIdx = attIndex;
      
      if ( Float.isNaN(prior) ) { // needs to be computed only once per branch - is same for all attributes (even regardless of missing values)
        prior = SplitCriteria.giniOverColumns(dist);
      }
      
      float negPosterior = - SplitCriteria.giniConditionedOnRows(dist);  // this is an updated dist
      if ( negPosterior > bestNegPosterior ) {
        bestNegPosterior = negPosterior;
      } else {
        throw new IllegalArgumentException("Very strange!");
      }

      val = prior - (-negPosterior); // we want the greatest reduction in entropy
      if ( val > 1e-2 ) {            // we allow some leeway here to compensate
        sensibleSplitFound = true;   // for imprecision in entropy computation
      }

    }  // feature by feature in window


    if ( sensibleSplitFound ) {

      m_Attribute = bestAttIdx;   // find best attribute
      m_SplitPoint = split;
      m_Prop = prop;
      prop = null; // can be GC'ed

//      long t = System.nanoTime();
      int belowTheSplitStartsAt = splitDataNew(m_Attribute, m_SplitPoint, sortedIndices, startAt, endAt, dist);
//      Benchmark.updateTime(System.nanoTime() - t);

      m_Successors = new FasterForest2Tree[2];  // dist.length now always == 2
      for (int i = 0; i < 2; i++) {
        FasterForest2Tree auxTree = new FasterForest2Tree(m_MotherForest, data /*, tempDistsL, tempDistsR, tempDistsOtherL, tempDistsOtherR, tempProps*/);

        // check if we're about to make an empty branch - this can happen with
        // nominal attributes with more than two categories (as of ver. 0.98)
        if (belowTheSplitStartsAt - startAt == 0) {
          // in this case, modify the chosenAttDists[i] so that it contains
          // the current, before-split class probabilities, properly normalized
          // by the number of instances (as we won't be able to normalize
          // after the split)
          //for (int j = 0; j < dist[i].length; j++)
          //  dist[i][j] = classProbs[j] / sortedIndicesLength;

          dist[i][0] = classProbs[0] / sortedIndicesLength;
          dist[i][1] = classProbs[1] / sortedIndicesLength;
        }

        if (i == 0) {   // before split
          auxTree.buildTree(sortedIndices, startAt, belowTheSplitStartsAt - 1,
                  dist[i], attIndicesWindow, depth + 1);
        } else {  // after split
          auxTree.buildTree(sortedIndices, belowTheSplitStartsAt, endAt,
                  dist[i], attIndicesWindow, depth + 1);
        }

        dist[i] = null;
        m_Successors[i] = auxTree;
      }
      sortedIndices = null;

    } else { // ------ make leaf --------

      m_Attribute = -1;

      // normalize by dividing with the number of instances (as of ver. 0.97)
      // unless leaf is empty - this can happen with splits on nominal attributes
      if ( sortedIndicesLength != 0 ) {
        //for (int c = 0; c < classProbs.length; c++) {
        //  classProbs[c] /= sortedIndicesLength;
        //}
        classProbs[0] /= sortedIndicesLength;
        classProbs[1] /= sortedIndicesLength;
      }
      m_ClassProbs = classProbs;
    }
    this.data = null; // dereference all pointers so data can be GC'd after tree is built
  }


  /**
   * Computes size of the tree.
   * @return the number of nodes
   */
  public int numNodes() {

    if (m_Attribute == -1) {
      return 1;
    } else {
      int size = 1;
      for (int i = 0; i < m_Successors.length; i++) {
        size += m_Successors[i].numNodes();
      }
      return size;
    }
  }


  /**
   * Splits instances into subsets; new for FastRF 0.99. Does not create new
   * arrays with split indices, but rather reorganizes the indices within the
   * supplied sortedIndices to conform with the split. Works only within given
   * boundaries. <p>
   *
   * Note: as of 0.99, all splits (incl. categorical) are always binary.
   *
   * @param att the attribute index
   * @param splitPoint the splitpoint for numeric attributes
   * @param sortedIndices the sorted indices of the whole set - gets overwritten!
   * @param startAt Inclusive, 0-based index. Does not touch anything before this value.
   * @param endAt  Inclusive, 0-based index. Does not touch anything after this value.
   * @param dist  dist[0] -> will have the counts of instances for the first branch.
   *              dist[1] -> will have the counts of instances for the second branch.
   *
   * @return the first index of the "below the split" instances
   */
  protected int splitDataNew(
          int att, float splitPoint,
          int[][] sortedIndices, int startAt, int endAt, float[][] dist ) {

    int j;
    // 0.99: we have binary splits also for nominal data
    int[] num = new int[2]; // how many instances go to each branch
    // we might possibly want to recycle this array for the whole tree
    int[] tempArr = new int[ endAt-startAt+1 ];
    Arrays.fill(dist[0], 0); Arrays.fill(dist[1], 0);


    int[] sortedIndicesAtt = sortedIndices[att];
    float[] dataValsA = data.vals[att];

    for (j = startAt; j <= endAt ; j++) {
      int inst = sortedIndicesAtt[j];
      int branch = ( dataValsA[inst] < splitPoint ) ? 0 : 1;
      data.whatGoesWhere[ inst ] = branch;
      dist[branch][data.instClassValues[inst]] += data.instWeights[inst];
      num[branch] += 1;
    } // end for instance by instance

    for (int a : data.attInSortedIndices) { // xxxxxxxxxx attr by attr

      // the first index of the sortedIndices in the above branch, and the first index in the below
      int startAbove = startAt, startBelow = 0; // always only 2 sub-branches, remember where second starts

      int[] sortedIndicesA = sortedIndices[a];

      // fill them with stuff by looking at goesWhere array
      for (j = startAt; j <= endAt; j++) {

        int inst = sortedIndicesA[j];
        int branch = data.whatGoesWhere[ inst ];  // can be only 0 or 1

        if ( branch==0 ) {
          sortedIndicesA[startAbove] = sortedIndicesA[j];
          startAbove++;
        } else {
          tempArr[startBelow] = sortedIndicesA[j];
          startBelow++;
        }
      }

      // now copy the tempArr into the sortedIndices, thus overwriting it
      System.arraycopy( tempArr, 0, sortedIndicesA, startAt+num[0], num[1] );



    } // xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx end for attr by attr

    return startAt+num[0]; // the first index of "below the split" instances
  }


  /**
   * Computes class distribution for an attribute. New in FastRF 0.99, main
   * changes:
   * <ul>
   *   <li> now reuses the temporary counting arrays (this.tempDists,
   *   this.tempDistsOthers) instead of creating/destroying arrays
   *   <li> does not create a new "dists" for each attribute it examines; instead
   *   it replaces the existing "dists" (supplied as a parameter) but only if the
   *   split is better than the previous best split
   *   <li> always creates binary splits, even for categorical variables; thus
   *   might give slightly different classification results than the old
   *   RandomForest
   * </ul>
   *
   * Edit: works only with numeric attributes
   *
   * @param propsBestAtt gets filled with relative sizes of branches (total = 1)
   * for the best examined attribute so far; updated ONLY if current attribute is
   * better that the previous best
   * @param distsBestAtt these are the contingency matrices for the best examined
   * attribute so far; updated ONLY if current attribute is better that the previous best
   * @param scoreBestAtt Checked against the score of the attToExamine to determine
   * if the propsBestAtt and distsBestAtt need to be updated.
   * @param attToExamine the attribute index (which one to examine, and change the above
   * matrices if the attribute is better than the previous one)
   * @param sortedIndicesOfAtt the sorted indices of the vals for the attToExamine.
   * @param startAt Index in sortedIndicesOfAtt; do not touch anything below this index.
   * @param endAt Index in sortedIndicesOfAtt; do not touch anything after this index.
   */
  protected float distributionSequentialAtt( float[] propsBestAtt, float[][] distsBestAtt,
                                              float scoreBestAtt, int attToExamine, int[] sortedIndicesOfAtt,
                                              int startAt, int endAt, float[] classProbs ) {

    float splitPoint = -Float.MAX_VALUE;

    // a contingency table of the split point vs class.
    //float[] distL = this.tempDistsL;
    //float[] distR = this.tempDistsR;
    //float[] currDistL = this.tempDistsOtherL;
    //float[] currDistR = this.tempDistsOtherR;

    float distL0;
    float distL1;
    float distR0;
    float distR1;
    float currDistL0;
    float currDistL1;
    float currDistR0;
    float currDistR1;

    // Copy the current class distribution
    //for (int i = 0; i < classProbs.length; ++i) {
    //  currDistR[i] = classProbs[i];
    //}
    currDistR0 = classProbs[0];
    currDistR1 = classProbs[1];

    //float[] props = this.tempProps;
    float props0;
    float props1;


    int i;
    //int sortedIndicesOfAttLength = endAt - startAt + 1;

    //Arrays.fill( currDistL, 0.0f );
    currDistL0 = currDistL1 = 0f;

    // find how many missing values we have for this attribute (they're always at the end)
    // update the distribution to the future second son
    int lastNonmissingValIdx = endAt;


    //copyDist(currDistL, distL);
    //copyDist(currDistR, distR);
    distL0 = currDistL0;
    distL1 = currDistL1;
    distR0 = currDistR0;
    distR1 = currDistR1;


    float currVal; // current value of splitting criterion
    float bestVal = -Float.MAX_VALUE; // best value of splitting criterion
    int bestI = 0; // the value of "i" BEFORE which the splitpoint is placed

    float[] dataValsAtt = data.vals[attToExamine]; // values of examined attribute

    for (i = startAt+1; i <= lastNonmissingValIdx; i++) {  // --- try all split points

      int inst = sortedIndicesOfAtt[i];
      int prevInst = sortedIndicesOfAtt[i-1];

      int prevInstClass = data.instClassValues[ prevInst ];
      double prevInstWeight = data.instWeights[ prevInst ];

      //currDistL[prevInstClass] += prevInstWeight;
      //currDistR[prevInstClass] -= prevInstWeight;
      if (prevInstClass==0) {
        currDistL0 += prevInstWeight;
        currDistR0 -= prevInstWeight;
      } else {
        currDistL1 += prevInstWeight;
        currDistR1 -= prevInstWeight;
      }

      // do not allow splitting between two instances with the same class or with the same value
      if (prevInstClass != data.instClassValues[inst] && dataValsAtt[inst] > dataValsAtt[prevInst] ) {
        currVal = -SplitCriteria.giniConditionedOnRowsLR2(currDistL0, currDistL1, currDistR0, currDistR1);
        if (currVal > bestVal) {
          bestVal = currVal;
          bestI = i;
        }
      }
    }                                             // ------- end trying split points

    /*
     * Determine the best split point:
     * bestI == 0 only if all instances had missing values, or there were
     * less than 2 instances; splitPoint will remain set as -Double.MAX_VALUE.
     * This is not really a useful split, as all of the instances are 'below'
     * the split line, but at least it's formally correct. And the dists[]
     * also has a default value set previously.
     */
    if ( bestI > startAt ) { // ...at least one valid splitpoint was found

      int instJustBeforeSplit = sortedIndicesOfAtt[bestI-1];
      int instJustAfterSplit = sortedIndicesOfAtt[bestI];
      splitPoint = ( dataValsAtt[ instJustAfterSplit ]
              + dataValsAtt[ instJustBeforeSplit ] ) / 2f;

      // now make the correct dist[] (for the best split point) from the
      // default dist[] (all instances in the second branch, by iterating
      // through instances until we reach bestI, and then stop.
      for ( int ii = startAt; ii < bestI; ii++ ) {
        int inst = sortedIndicesOfAtt[ii];
        int instClass = data.instClassValues[inst];
        double instWeight = data.instWeights[inst];

        //distL[instClass] += instWeight;
        //distR[instClass] -= instWeight;
        if (instClass==0) {
          distL0 += instWeight;
          distR0 -= instWeight;
        } else {
          distL1 += instWeight;
          distR1 -= instWeight;
        }
      }
    }

    // compute total weights for each branch (= props)
    // again, we reuse the tempProps of the tree not to create/destroy new arrays
    //countsToFreqsLR2(distL0, distL1, distR0, distR1, props);  // props gets overwritten, previous contents don't matters
    props0 = distL0 + distL1;
    props1 = distR0 + distR1;

    float propsSum = props0 + props1;
    if (propsSum == 0.0f) {
      props0 = 0.5f;
      props1 = 0.5f;
    } else {
      props0 /= propsSum;
      props1 /= propsSum;
    }

    // distribute *counts* of instances with missing values using the "props"
    // start 1 after the non-missing val (if there is anything)
    // removed support for missing values
    //for (i = lastNonmissingValIdx + 1; i <= endAt; ++i) {
    //  int inst = sortedIndicesOfAtt[i];
    //  int instClass = data.instClassValues[inst];
    //  double instWeight = data.instWeights[inst];
    //
    //  dist[ 0 ][ instClass ] += props[ 0 ] * instWeight ;
    //  dist[ 1 ][ instClass ] += props[ 1 ] * instWeight ;
    //}

    // update the distribution after split and best split point
    // but ONLY if better than the previous one -- we need to recalculate the
    // entropy (because this changes after redistributing the instances with
    // missing values in the current attribute). Also, for categorical variables
    // it was not calculated before.
    float curScore = -SplitCriteria.giniConditionedOnRowsLR2(distL0, distL1, distR0, distR1);
    if ( curScore > scoreBestAtt && splitPoint > -Double.MAX_VALUE ) {  // overwrite the "distsBestAtt" and "propsBestAtt" with current values

      //copyDist(distL, distsBestAtt[0]);
      //copyDist(distR, distsBestAtt[1]);
      distsBestAtt[0][0] = distL0;
      distsBestAtt[0][1] = distL1;
      distsBestAtt[1][0] = distR0;
      distsBestAtt[1][1] = distR1;

      propsBestAtt[0] = props0;
      propsBestAtt[1] = props1;

      return splitPoint;
    } else {
      // returns a NaN instead of the splitpoint if the attribute was not better than a previous one.
      return Float.NaN;
    }
  }

  /**
   * Normalizes branch sizes so they contain frequencies (stored in "props")
   * instead of counts (stored in "dist"). Creates a new double[] which it 
   * returns.
   */  
  protected static float[] countsToFreqs( float[][] dist ) {

    float[] props = new float[dist.length];
    
    for (int k = 0; k < props.length; k++) {
      props[k] = FastRfUtils.sum(dist[k]);
    }
    if (Utils.eq(FastRfUtils.sum(props), 0)) {
      Arrays.fill(props, 1.0f / (float) props.length);
    } else {
      FastRfUtils.normalize(props);
    }
    return props;
  }

  /**
   * Normalizes branch sizes so they contain frequencies (stored in "props")
   * instead of counts (stored in "dist"). <p>
   * 
   * Overwrites the supplied "props"! <p>
   * 
   * props.length must be == dist.length == 2.
   */  
  protected static void countsToFreqs( float[][] dist, float[] props ) {
    
    for (int k = 0; k < props.length; k++) {
      props[k] = FastRfUtils.sum(dist[k]);
    }

    if (Utils.eq(FastRfUtils.sum(props), 0.0)) {
      Arrays.fill(props, 1.0f / (float) props.length);
    } else {
      FastRfUtils.normalize(props);
    }

  }

  protected static void countsToFreqsLR( float[] distL, float[] distR, float[] props ) {
    props[0] = FastRfUtils.sum(distL);
    props[1] = FastRfUtils.sum(distR);

    if (Utils.eq(FastRfUtils.sum(props), 0.0)) {
      Arrays.fill(props, 1.0f / (float) props.length);
    } else {
      FastRfUtils.normalize(props);
    }
  }

  protected static void countsToFreqsLR2( float distL0, float distL1, float distR0, float distR1, float[] props ) {
    props[0] = distL0 + distL1;
    props[1] = distR0 + distR1;

    float sum = props[0] + props[1];
    if (sum == 0.0f) {
      props[0] = 0.5f;
      props[1] = 0.5f;
    } else {
      props[0] /= sum;
      props[1] /= sum;
    }
  }

  
  /**
   * Makes a copy of a "dists" array, which is a 2 x numClasses array. 
   * 
   * @param distFrom
   * @param distTo Gets overwritten.
   */
  protected static void copyDists( double[][] distFrom, double[][] distTo ) {
    for ( int i = 0; i < distFrom[0].length; i++ ) {
      distTo[0][i] = distFrom[0][i];
    }
    for ( int i = 0; i < distFrom[1].length; i++ ) {
      distTo[1][i] = distFrom[1][i];
    }
  }

  protected static void copyDist( float[] distFrom, float[] distTo ) {
    for (int i = 0; i < distFrom.length; i++) {
      distTo[i] = distFrom[i];
    }
  }
  
  
  /**
   * Main method for this class.
   * 
   * @param argv the commandline parameters
   */
  public static void main(String[] argv) {
    runClassifier(new FasterForest2Tree(), argv);
  }



  @Override
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 2.0$");
  }


  /**
   * Convert tree to leaner version.
   * Dismantles this tree in the process.
   */
  public FasterTree toLightVersion() {
    int attribute = m_Attribute;
    if (attribute < 0) {
      attribute = -1;
    }
    boolean isLeaf = (attribute == -1);

    FasterTree leftChild = null;
    FasterTree rightChild = null;
    if (!isLeaf) {
      leftChild = m_Successors[0].toLightVersion();
      m_Successors[0] = null;  // to allow gc
      rightChild = m_Successors[1].toLightVersion();
      m_Successors[1] = null;     // to allow gc
      m_Successors = null;     // to allow gc
    }

    double[] classProbs = (isLeaf) ? FastRfUtils.toDoubles2(m_ClassProbs) : null;

    FasterTree res = new FasterTree(leftChild, rightChild, attribute, m_SplitPoint, classProbs);

    return res;
  }

  
}

