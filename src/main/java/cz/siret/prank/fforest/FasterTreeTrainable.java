package cz.siret.prank.fforest;

import weka.core.Utils;

import java.util.Arrays;

/**
 * Extended version of FasterForest used in training.
 */
public class FasterTreeTrainable extends FasterTree {

    /** for serialization */
    static final long serialVersionUID = -9136056750085906361L;

    /**
     * For access to parameters of the RF (k, or maxDepth).
     */
    protected FasterForest m_MotherForest;

    /** The dataset used for training. */
    protected transient DataCache data = null;

    /**
     * Since 0.99: holds references to temporary arrays re-used by all nodes
     * in the tree, used while calculating the "props" for various attributes in
     * distributionSequentialAtt(). This is meant to avoid frequent
     * creating/destroying of these arrays.
     */
    protected transient double[] tempProps;

    /**
     * Since 0.99: holds references to temporary arrays re-used by all nodes
     * in the tree, used while calculating the "dists" for various attributes
     * in distributionSequentialAtt(). This is meant to avoid frequent
     * creating/destroying of these arrays.
     */
    protected transient double[][] tempDists;
    protected transient double[][] tempDistsOther;

    /**
     * created in run(), one for each root tree
     */
    private int[] tempIndices = null;


    /**
     * Get the value of K.
     *
     * @return Value of K.
     */
    public final int getKValue() {
        return m_MotherForest.m_KValue;
    }


    /**
     * Get the maximum depth of the tree, 0 for unlimited.
     *
     * @return 		the maximum depth.
     */
    public final int getMaxDepth() {
        return m_MotherForest.m_MaxDepth;
    }

    public FasterTreeTrainable() {
    }

    public FasterTreeTrainable(FasterTreeTrainable from) {
        this.m_MotherForest = from.m_MotherForest;
        this.data = from.data;
        this.tempDists = from.tempDists;
        this.tempDistsOther = from.tempDistsOther;
        this.tempProps = from.tempProps;
        this.tempIndices = from.tempIndices;
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
                             double[] classProbs,
                             // boolean debug,
                             int[] attIndicesWindow,
                             int depth)  {

        //m_Debug = debug;
        int sortedIndicesLength = endAt - startAt + 1;

        // Check if node doesn't contain enough instances or is pure
        // or maximum depth reached, make leaf.
        if ( ( sortedIndicesLength < Math.max(2, getMinNum()) )  // small
            || Utils.eq( classProbs[Utils.maxIndex(classProbs)], Utils.sum(classProbs) )       // pure
            || ( (getMaxDepth() > 0)  &&  (depth >= getMaxDepth()) )                           // deep
            ) {
            m_Attribute = -1;  // indicates leaf (no useful attribute to split on)

            // normalize by dividing with the number of instances (as of ver. 0.97)
            // unless leaf is empty - this can happen with splits on nominal
            // attributes with more than two categories
            if ( sortedIndicesLength != 0 )
                for (int c = 0; c < classProbs.length; c++) {
                    classProbs[c] /= sortedIndicesLength;
                }
            m_ClassProbs = classProbs;
            this.data = null;
            return;
        } // (leaf making)

        // new 0.99: all the following are for the best attribute only! they're updated while sequentially through the attributes
        double val = Double.NaN; // value of splitting criterion
        double[][] dist = new double[2][data.numClasses];  // class distributions (contingency table), indexed first by branch, then by class
        double[] prop = new double[2]; // the branch sizes (as fraction)
        double split = Double.NaN;  // split point

        // Investigate K random attributes
        int attIndex = 0;
        int windowSize = attIndicesWindow.length;
        int k = getKValue();
        boolean sensibleSplitFound = false;
        double prior = Double.NaN;
        double bestNegPosterior = -Double.MAX_VALUE;
        int bestAttIdx = -1;

        while ((windowSize > 0) && (k-- > 0 || !sensibleSplitFound ) ) {

            int chosenIndex = data.reusableRandomGenerator.nextInt(windowSize);
            attIndex = attIndicesWindow[chosenIndex];

            // shift chosen attIndex out of window
            attIndicesWindow[chosenIndex] = attIndicesWindow[windowSize - 1];
            attIndicesWindow[windowSize - 1] = attIndex;
            windowSize--;

            // new: 0.99
            double candidateSplit = distributionSequentialAtt( prop, dist,
                bestNegPosterior, attIndex,
                sortedIndices[attIndex], startAt, endAt );

            if ( Double.isNaN(candidateSplit) ) {
                continue;  // we did not improve over a previous attribute! "dist" is unchanged from before
            }
            // by this point we know we have an improvement, so we keep the new split point
            split = candidateSplit;
            bestAttIdx = attIndex;

            if ( Double.isNaN(prior) ) { // needs to be computed only once per branch - is same for all attributes (even regardless of missing values)
                prior = SplitCriteria.entropyOverColumns(dist);
            }

            double negPosterior = - SplitCriteria.entropyConditionedOnRows(dist);  // this is an updated dist
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

        prop = null; // can be GC'ed
        if ( sensibleSplitFound ) {

            m_Attribute = bestAttIdx;   // find best attribute
            m_SplitPoint = split;
            //m_Prop = prop;



            //int[][][] subsetIndices =
            //        new int[dist.length][data.numAttributes][];
            //splitData( subsetIndices, m_Attribute,
            //        m_SplitPoint, sortedIndices );
            //int numInstancesBeforeSplit = sortedIndices[0].length;

            int belowTheSplitStartsAt = splitDataNew(  m_Attribute, m_SplitPoint, sortedIndices, startAt, endAt );


            //m_Successors = new FasterTree[dist.length];  // dist.length now always == 2
            sucessorLeft = new FasterTreeTrainable(this);
            sucessorRight = new FasterTreeTrainable(this);
            FasterTreeTrainable tree = (FasterTreeTrainable) sucessorLeft;
            for (int i = 0; i < dist.length; i++) {    // 0..1
                if (i==1)
                    tree = (FasterTreeTrainable) sucessorRight;



                // check if we're about to make an empty branch - this can happen with
                // nominal attributes with more than two categories (as of ver. 0.98)
                if ( belowTheSplitStartsAt - startAt == 0  ) {
                    // in this case, modify the chosenAttDists[i] so that it contains
                    // the current, before-split class probabilities, properly normalized
                    // by the number of instances (as we won't be able to normalize
                    // after the split)
                    for ( int j = 0; j < dist[i].length; j++ )
                        dist[i][j] = classProbs[j] / sortedIndicesLength;
                }

                if ( i == 0 ) {   // before split
                    tree.buildTree(sortedIndices, startAt, belowTheSplitStartsAt - 1,
                        dist[i], attIndicesWindow, depth + 1);
                } else {  // after split
                    tree.buildTree(sortedIndices, belowTheSplitStartsAt, endAt,
                        dist[i], attIndicesWindow, depth + 1);
                }


                dist[i] = null;

            }
            sortedIndices = null;


        } else { // ------ make leaf --------

            m_Attribute = -1;

            // normalize by dividing with the number of instances (as of ver. 0.97)
            // unless leaf is empty - this can happen with splits on nominal attributes
//            if ( sortedIndicesLength != 0 )
            for (int c = 0; c < classProbs.length; c++) {
                classProbs[c] /= sortedIndicesLength;
            }

            m_ClassProbs = classProbs;

        }

        this.data = null; // dereference all pointers so data can be GC'd after tree is built

    }



//  /**
//   * Splits instances into subsets. Not used anymore in 0.99. This is a
//   * derivative of the splitData function from "weka.classifiers.trees.RandomTree",
//   * with the following changes: <p>
//   *
//   * - When handling instances with missing values in attribute chosen for the
//   * split, the FasterForest2Tree assignes the instance to one of the branches at
//   * random, with bigger branches having a higher probability of getting the
//   * instance. <p>
//   *
//   * - When splitting sortedIndices into two or more subsetIndices,
//   * FasterForest2Tree checks whether an instance's split attribute value was above
//   * splitpoint only once per instances, and stores result into the DataCache's
//   * whatGoesWhere field, which is then read in splitting subsetIndices. <p>
//   *
//   * As a consequence of the above points, the exact branch sizes (even with
//   * instances having unknowns in the split attribute) are known in advance so
//   * subsetIndices arrays don't have to be 'resized' (i.e. a new shorter copy
//   * of each one created and the old one GCed). <p>
//   *
//   * @param subsetIndices the sorted indices of the subset
//   * @param att the attribute index
//   * @param splitPoint the splitpoint for numeric attributes
//   * @param sortedIndices the sorted indices of the whole set
//   */
//  protected void splitData( int[][][] subsetIndices,
//          int att, double splitPoint,
//          int[][] sortedIndices ) {
//
//    Random random = data.reusableRandomGenerator;
//    int j;
//    // 0.99: we have binary splits also for nominal data
//    int[] num = new int[2]; // how many instances go to each branch
//
//    if ( data.isAttrNominal(att) ) { // ============================ if nominal
//
//      for (j = 0; j < sortedIndices[att].length; j++) {
//
//        int inst = sortedIndices[att][j];
//
//        if ( data.isValueMissing(att, inst) ) { // ---------- has missing value
//
//          // decide where to put this instance randomly, with bigger branches
//          // getting a higher chance
//          double rn = random.nextDouble();
//          int myBranch = -1;
//          for (int k = 0; k < m_Prop.length; k++) {
//            rn -= m_Prop[k];
//            if ( (rn <= 0) || k == (m_Prop.length-1) ) {
//              myBranch = k;
//              break;
//            }
//          }
//
//          data.whatGoesWhere[ inst ] = myBranch;
//          num[myBranch]++;
//
//        } else { // ----------------------------- does not have missing value
//
//          // if it matches the category to "split out", put above split
//          // all other categories go below split
//          int subset = ( data.vals[att][inst] == splitPoint ) ? 0 : 1;
//          data.whatGoesWhere[ inst ] = subset;
//          num[subset]++;
//
//        } // --------------------------------------- end if has missing value
//
//      }
//
//    } else { // =================================================== if numeric
//
//      num = new int[2];
//
//      for (j = 0; j < sortedIndices[att].length; j++) {
//
//        int inst = sortedIndices[att][j];
//
//        //Instance inst = data.instance(sortedIndices[att][j]);
//
//        if ( data.isValueMissing(att, inst) ) { // ---------- has missing value
//
//          // decide if instance goes into subset 0 or 1 randomly,
//          // with bigger subsets having a greater probability of getting
//          // the instance assigned to them
//          // instances with missing values get processed LAST (sort order)
//          // so branch sizes are known by now (and stored in m_Prop)
//          double rn = random.nextDouble();
//          int branch = ( rn > m_Prop[0] ) ? 1 : 0;
//          data.whatGoesWhere[ inst ] = branch;
//          num[ branch ]++;
//
//        } else { // ----------------------------- does not have missing value
//
//          int branch = ( data.vals[att][inst] < splitPoint ) ? 0 : 1;
//
//          data.whatGoesWhere[ inst ] = branch;
//          num[ branch ]++;
//
//        } // --------------------------------------- end if has missing value
//
//      } // end for instance by instance
//
//    }  // ============================================ end if nominal / numeric
//
//
//
//
//    // create the new subset (branch) arrays of correct size -- as of 0.99, not anymore
//    for (int a = 0; a < data.numAttributes; a++) {
//      if ( a == data.classIndex )
//        continue;   // no need to sort this one
//      for (int branch = 0; branch < num.length; branch++) {
//        subsetIndices[branch][a] = new int[num[branch]];
//      }
//    }
//
//    for (int a = 0; a < data.numAttributes; a++) { // xxxxxxxxxx attr by attr
//
//      if (a == data.classIndex)
//        continue;
//      for (int branch = 0; branch < num.length; branch++) {
//        num[branch] = 0;
//      }
//
//      // fill them with stuff by looking at goesWhere array
//      for (j = 0; j < sortedIndices[ a ].length; j++) {
//
//        int inst = sortedIndices[ a ][j];
//        int branch = data.whatGoesWhere[ inst ];  // can be 0 or 1
//
//        subsetIndices[ branch ][ a ][ num[branch] ] = sortedIndices[a][j];
//        num[branch]++;
//
//      }
//
//    } // xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx end for attr by attr
//
//  }





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
     *
     * @return the first index of the "below the split" instances
     */
    protected int splitDataNew(int att, double splitPoint, int[][] sortedIndices, int startAt, int endAt ) {

        //Random random = data.reusableRandomGenerator;
        int j;
        // 0.99: we have binary splits also for nominal data
        //int[] num; //= new int[2]; // how many instances go to each branch

        // we might possibly want to recycle this array for the whole tree
        int[] tempArr = tempIndices;

        //num = new int[2];

        int num0 = 0;

        float[] dataValsAtt = data.vals[att];
        int[] sortIndAtt = sortedIndices[att];
        int[] whatGoesWhere = data.whatGoesWhere;
        int numAttributes = data.numAttributes;
        int classIndex = data.classIndex;

        for (j = startAt; j <= endAt ; j++) {

            int inst = sortIndAtt[j];

            //Instance inst = data.instance(sortedIndices[att][j]);

            int branch = ( dataValsAtt[inst] < splitPoint ) ? 0 : 1;

            whatGoesWhere[ inst ] = branch;
            if (branch == 0) {
                num0++;
            }

        }

        for (int a = 0; a < numAttributes; a++) { // xxxxxxxxxx attr by attr

            if (a == classIndex)
                continue;

            // the first index of the sortedIndices in the above branch, and the first index in the below
            int startAbove = 0, startBelow = num0; // always only 2 sub-branches, remember where second starts

            // TODO make faster custom fill
            Arrays.fill(tempArr, 0, endAt-startAt+1, 0);

            //for (int branch = 0; branch < num.length; branch++) {
            //  num[branch] = 0;
            //}

            int[] sortedIndicesA = sortedIndices[a];

            // fill them with stuff by looking at goesWhere array
            for (j = startAt; j <= endAt; j++) {

                int inst = sortedIndicesA[j];
                int branch = whatGoesWhere[ inst ];  // can be only 0 or 1

                if ( branch==0 ) {
                    tempArr[ startAbove ] = inst;
                    startAbove++;
                } else {
                    tempArr[ startBelow ] = inst;
                    startBelow++;
                }

                //subsetIndices[ branch == 0 ? startAbove :  ][ a ][ num[branch] ] = sortedIndices[a][j];
                //num[branch]++;

            }

            // now copy the tempArr into the sortedIndices, thus overwriting it
            System.arraycopy( tempArr, 0, sortedIndicesA, startAt, endAt-startAt+1 );

        } // xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx end for attr by attr

        return startAt+num0; // the first index of "below the split" instances

    }



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
     * @return the computed class distribution
     */
    public double[] distributionForInstanceInDataCache(DataCache data, int instIdx) {

        if (m_Attribute != -1) {  // ============================ node is not a leaf

            if ( data.vals[m_Attribute][instIdx] < m_SplitPoint) {
                return ((FasterTreeTrainable)sucessorLeft).distributionForInstanceInDataCache(data, instIdx);
            } else {
                return ((FasterTreeTrainable)sucessorRight).distributionForInstanceInDataCache(data, instIdx);
            }

        } else { // =============================================== node is a leaf

            return m_ClassProbs;

        }

    }


    /**
     * Computes class distribution for an attribute. Not used anymore in 0.99.
     * Based on the splitData function from "weka.classifiers.trees.RandomTree",
     * with the following changes:<ul>
     *
     * <li>entropy pre-split is not computed at this point as the only thing
     *     relevant for the (comparative) goodness of a split is entropy after splitting
     * <li>dist[][] is now computed only after the split point has been found,
     *     and not updated continually by copying from currDist
     * <li>also, in Weka's RandomTree it was possible to create a split 'in the
     *     middle' of instance 0, which would result in empty nodes after the
     *     split; this is now fixed
     * <li>instance 0 is now generally skipped when looking for split points,
     *     as the split point 'before instance 0' is not sensible; in versions
     *     prior to 0.96 this change introduced a bug where attributes with
     *     all missing values had their dists computed wrongly, which might
     *     result in useless (but harmless) branches being added to the tree
     * </ul>
     *
     * @param props gets filled with relative sizes of branches (total = 1), indexed
     * first per attribute
     * @param dists these are the contingency matrices, indexed first per attribute
     * @param att the attribute index (which one to change)
     * @param sortedIndices the sorted indices of the vals
     */
    protected double distribution( double[][] props, double[][][] dists,
                                   int att, int[] sortedIndices ) {

        double splitPoint = -Double.MAX_VALUE;
        double[][] dist = null;  // a contingency table of the split point vs class
        int i;

//    if ( data.isAttrNominal(att) ) { // ====================== nominal attributes
//
//      dist = new double[data.attNumVals[att]][data.numClasses];
//      for (i = 0; i < sortedIndices.length; i++) {
//        int inst = sortedIndices[i];
//        if ( data.isValueMissing(att, inst) )
//          break;
//        dist[ (int)data.vals[att][inst] ][ data.instClassValues[inst] ] += data.instWeights[inst];
//      }
//
//      splitPoint = 0; // signals we've found a sensible split point; by
//                      // definition, a split on a nominal attribute is sensible
//
//    } else { // ============================================ numeric attributes

        double[][] currDist = new double[2][data.numClasses];
        dist = new double[2][data.numClasses];

        //begin with moving all instances into second subset
        for (int inst : sortedIndices) {
            if (data.isValueMissing(att, inst))
                break;
            currDist[1][data.instClassValues[inst]] += data.instWeights[inst];
        }
        copyDists(currDist, dist);
        //for (int j = 0; j < currDist.length; j++)
        //  System.arraycopy(currDist[j], 0, dist[j], 0, dist[j].length);

        double currVal = -Double.MAX_VALUE; // current value of splitting criterion
        double bestVal = -Double.MAX_VALUE; // best value of splitting criterion
        int bestI = 0; // the value of "i" BEFORE which the splitpoint is placed

        for (i = 1; i < sortedIndices.length; i++) {  // --- try all split points

            int inst = sortedIndices[i];
            if ( data.isValueMissing(att, inst) )
                break;

            int prevInst = sortedIndices[i-1];

            currDist[0][ data.instClassValues[ prevInst ] ]
                += data.instWeights[ prevInst ] ;
            currDist[1][ data.instClassValues[ prevInst ] ]
                -= data.instWeights[ prevInst ] ;

            // do not allow splitting between two instances with the same value
            if ( data.vals[att][inst] > data.vals[att][prevInst] ) {

                // we want the lowest impurity after split; at this point, we don't
                // really care what we've had before spliting
                currVal = -SplitCriteria.entropyConditionedOnRows(currDist);

                if (currVal > bestVal) {
                    bestVal = currVal;
                    bestI = i;
                }

            }

        }                                             // ------- end split points

      /*
       * Determine the best split point:
       * bestI == 0 only if all instances had missing values, or there were
       * less than 2 instances; splitPoint will remain set as -Double.MAX_VALUE.
       * This is not really a useful split, as all of the instances are 'below'
       * the split line, but at least it's formally correct. And the dists[]
       * also has a default value set previously.
       */
        if ( bestI > 0 ) { // ...at least one valid splitpoint was found

            int instJustBeforeSplit = sortedIndices[bestI-1];
            int instJustAfterSplit = sortedIndices[bestI];
            splitPoint = ( data.vals[ att ][ instJustAfterSplit ]
                + data.vals[ att ][ instJustBeforeSplit ] ) / 2.0;

            // Now make the correct dist[] from the default dist[] (all instances
            // in the second branch, by iterating through instances until we reach
            // bestI, and then stop.
            for ( int ii = 0; ii < bestI; ii++ ) {
                int inst = sortedIndices[ii];
                dist[0][ data.instClassValues[ inst ] ] += data.instWeights[ inst ] ;
                dist[1][ data.instClassValues[ inst ] ] -= data.instWeights[ inst ] ;
            }

        }

//    } // ================================================== nominal or numeric?

        // compute total weights for each branch (= props)
        props[att] = countsToFreqs(dist);

        // distribute counts of instances with missing values

        // ver 0.96 - check for special case when *all* instances have missing vals
//    if ( data.isValueMissing(att, sortedIndices[0]) )
//      i = 0;

        while (i < sortedIndices.length) {
            int inst = sortedIndices[i];
            for (int branch = 0; branch < dist.length; branch++) {
                dist[ branch ][ data.instClassValues[inst] ]
                    += props[ att ][ branch ] * data.instWeights[ inst ] ;
            }
            i++;
        }

        // return distribution after split and best split point
        dists[att] = dist;
        return splitPoint;

    }


    public static void fill0(double[] a, double[] b) {
        for (int i = 0, len = a.length; i != len; i++) {
            a[i] = 0d;
            b[i] = 0d;
         }
    }

    public static void fill0(double[] a, double[] b, double[] c, double[] d) {
        for (int i = 0, len = a.length; i < len; i++) {
            a[i] = 0d;
            b[i] = 0d;
            c[i] = 0d;
            d[i] = 0d;
        }
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
    protected final double distributionSequentialAtt(double[] propsBestAtt, double[][] distsBestAtt,
                                                     double scoreBestAtt, int attToExamine, int[] sortedIndicesOfAtt, int startAt, int endAt) {

        double splitPoint = -Double.MAX_VALUE;

        // a contingency table of the split point vs class.
        double[][] dist = this.tempDists;
        double[] dist0 = dist[0];
        double[] dist1 = dist[1];

        double[][] currDist = this.tempDistsOther;
        double[] currDist0 = currDist[0];
        double[] currDist1 = currDist[1];

        fill0(dist0, dist1, currDist0, currDist1);

        //double[][] dist = new double[2][data.numClasses];
        //double[][] currDist = new double[2][data.numClasses];

        float[] attributeValues = data.vals[attToExamine];
        int[] instClassValues = data.instClassValues;
        double[] instWeights = data.instWeights;


        int i;
//        int sortedIndicesOfAttLength = endAt - startAt + 1;

        // find how many missing values we have for this attribute (they're always at the end)
        int lastNonmissingValIdx = endAt;
//        for (int j = endAt; j >= startAt; j-- ) {
//            if ( data.isValueMissing(attToExamine, sortedIndicesOfAtt[j]) ) {
//                lastNonmissingValIdx = j-1;
//            } else {
//                break;
//            }
//        }
//        if ( lastNonmissingValIdx < startAt ) {  // only missing values in this feature??
//            return Double.NaN; // we cannot split on it
//        }


        // re-use the 2 x nClass temporary arrays created when tree was initialized
        //Arrays.fill( dist[0], 0.0 );
        //Arrays.fill( dist[1], 0.0 );

        // begin with moving all instances into second subset ("below split")
        for (int j = startAt; j <= lastNonmissingValIdx; j++) {
            int inst = sortedIndicesOfAtt[j];
            currDist1[instClassValues[inst]] += instWeights[inst];
        }
        copyDists(currDist, dist);

        double currVal; // = -Double.MAX_VALUE; // current value of splitting criterion
        double bestVal = -Double.MAX_VALUE; // best value of splitting criterion
        int bestI = 0; // the value of "i" BEFORE which the splitpoint is placed

        for (i = startAt + 1; i <= lastNonmissingValIdx; i++) {  // --- try all split points

            int inst = sortedIndicesOfAtt[i];
            int prevInst = sortedIndicesOfAtt[i - 1];

            int classValuePI = instClassValues[prevInst];
            double weightPI = instWeights[prevInst];

            currDist0[classValuePI] += weightPI;
            currDist1[classValuePI] -= weightPI;

            // do not allow splitting between two instances with the same value
            if (attributeValues[inst] > attributeValues[prevInst]) {

                // we want the lowest impurity after split; at this point, we don't
                // really care what we've had before spliting
                currVal = -SplitCriteria.entropyConditionedOnRows(currDist);

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
        if (bestI > startAt) { // ...at least one valid splitpoint was found

            int instJustBeforeSplit = sortedIndicesOfAtt[bestI - 1];
            int instJustAfterSplit = sortedIndicesOfAtt[bestI];
            splitPoint = (attributeValues[instJustAfterSplit] + attributeValues[instJustBeforeSplit]) / 2.0;

            // now make the correct dist[] (for the best split point) from the
            // default dist[] (all instances in the second branch, by iterating
            // through instances until we reach bestI, and then stop.
            for (int ii = startAt; ii < bestI; ii++) {
                int inst = sortedIndicesOfAtt[ii];

                int classValueI = instClassValues[inst];
                double weightI = instWeights[inst];

                dist0[classValueI] += weightI;
                dist1[classValueI] -= weightI;
            }

        }

        //  } // ================================================== nominal or numeric?


        // compute total weights for each branch (= props)
        // again, we reuse the tempProps of the tree not to create/destroy new arrays
        double[] props = this.tempProps;
        countsToFreqs(dist, props);  // props gets overwritten, previous contents don't matters

        //double props0 = props[0];
        //double props1 = props[1];
        // distribute *counts* of instances with missing values using the "props"
        //i = lastNonmissingValIdx + 1; /// start 1 after the non-missing val (if there is anything)
        //while (i <= endAt) {
        //    int inst = sortedIndicesOfAtt[i];
        //
        //    int classValuesI = instClassValues[inst];
        //    double weightsI = instWeights[inst];
        //
        //    dist0[classValuesI] += props0 * weightsI;
        //    dist1[classValuesI] += props1 * weightsI;
        //    i++;
        //}

        // update the distribution after split and best split point
        // but ONLY if better than the previous one -- we need to recalculate the
        // entropy (because this changes after redistributing the instances with
        // missing values in the current attribute). Also, for categorical variables
        // it was not calculated before.
        double curScore = -SplitCriteria.entropyConditionedOnRows(dist);
        if (curScore > scoreBestAtt && splitPoint > -Double.MAX_VALUE) {  // overwrite the "distsBestAtt" and "propsBestAtt" with current values
            copyDists(dist, distsBestAtt);
            System.arraycopy(props, 0, propsBestAtt, 0, props.length);
            return splitPoint;
        } else {
            // returns a NaN instead of the splitpoint if the attribute was not better than a previous one.
            return Double.NaN;
        }


    }


    /**
     * Normalizes branch sizes so they contain frequencies (stored in "props")
     * instead of counts (stored in "dist"). Creates a new double[] which it
     * returns.
     */
    protected static double[] countsToFreqs( double[][] dist ) {

        double[] props = new double[dist.length];

        for (int k = 0; k < props.length; k++) {
            props[k] = Utils.sum(dist[k]);
        }
        if (Utils.eq(Utils.sum(props), 0)) {
            Arrays.fill(props, 1.0 / (double) props.length);
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
     * props.length must be == dist.length.
     */
    protected static void countsToFreqs( double[][] dist, double[] props ) {

        for (int k = 0; k < props.length; k++) {
            props[k] = Utils.sum(dist[k]);
        }
        if (Utils.eq(Utils.sum(props), 0)) {
            Arrays.fill(props, 1.0 / (double) props.length);
        } else {
            FastRfUtils.normalize(props);
        }

    }


    /**
     * Makes a copy of a "dists" array, which is a 2 x numClasses array.
     *
     * @param distFrom
     * @param distTo Gets overwritten.
     */
    protected static void copyDists( double[][] distFrom, double[][] distTo ) {
        for ( int i = 0, len = distFrom[0].length; i < len; i++ ) {
            distTo[0][i] = distFrom[0][i];
            distTo[1][i] = distFrom[1][i];
        }
    }

//    protected static void copyDists( double[][] distFrom, double[][] distTo ) {
//        for ( int i = 0; i < distFrom[0].length; i++ ) {
//            distTo[0][i] = distFrom[0][i];
//        }
//        for ( int i = 0; i < distFrom[1].length; i++ ) {
//            distTo[1][i] = distFrom[1][i];
//        }
//    }


    /**
     * Builds classifier. Makes the initial call to the recursive buildTree
     * function. The name "run()" is used to support multithreading via an
     * ExecutorService. <p>
     *
     * The "data" field of the FasterForest2Tree should contain a
     * reference to a DataCache prior to calling this function, and that
     * DataCache should have the "reusableRandomGenerator" field initialized.
     * The FastRfBagging class normally takes care of this before invoking this
     * function.
     */
    public void buildRootTree(DataCache data) {
        this.data = data;
        int n = data.numInstances;

        // compute initial class counts
        double[] classProbs = new double[data.numClasses];
        for (int i = 0; i < n; i++) {
            classProbs[data.instClassValues[i]] += data.instWeights[i];
        }

        // create the attribute indices window - skip class
        int[] attIndicesWindow = new int[data.numAttributes - 1];
        int j = 0;
        for (int i = 0; i < attIndicesWindow.length; i++) {
            if (j == data.classIndex)
                j++; // do not include the class
            attIndicesWindow[i] = j++;
        }

        // prepare the DataCache by:
        // ... creating an array for the whatGoesWhere field of the data
        // ... creating the sortedIndices
        // TODO recycle this array
        data.whatGoesWhere = new int[ data.inBag.length ];
        data.createInBagSortedIndices();

        // TODO recycle this array
        tempIndices = new int[n];

        buildTree(data.sortedIndices, 0, data.sortedIndices[0].length-1,
            classProbs, attIndicesWindow, 0);

        this.data = null;

    }


}
