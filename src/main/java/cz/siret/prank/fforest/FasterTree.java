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
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    FasterForest2Tree.java
 *    Copyright (C) 2001 University of Waikato, Hamilton, NZ (original code,
 *      RandomTree.java)
 *    Copyright (C) 2013 Fran Supek (adapted code)
 */

package cz.siret.prank.fforest;

import weka.classifiers.Classifier;
import weka.core.*;
import weka.core.Capabilities.Capability;

import java.io.Serializable;


/**
 * Based on the "weka.classifiers.trees.RandomTree" class, revision 1.19,
 * by Eibe Frank and Richard Kirkby, with major modifications made to improve
 * the speed of classifier training.
 * 
 * Please refer to the Javadoc of buildTree, splitData and distribution
 * function, as well as the changelog.txt, for the details of changes to 
 * FasterTreeTrainable.
 * 
 * This class should be used only from within the FasterForest classifier.
 * 
 * @author Eibe Frank (eibe@cs.waikato.ac.nz) - original code
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz) - original code
 * @author Fran Supek (fran.supek[AT]irb.hr) - adapted code
 * @version $Revision: 0.99$
 */
public class FasterTree
        // extends AbstractClassifier
        implements Classifier, Serializable, Cloneable, CapabilitiesHandler, WeightedInstancesHandler {

  /** for serialization */
  static final long serialVersionUID = 8934314652175299376L;

  /** Minimum number of instances for leaf. */
  protected static final int m_MinNum = 1;


  /** The subtrees appended to this tree (node). */
  protected FasterTree sucessorLeft;
  protected FasterTree sucessorRight;

  /** The attribute to split on. */
  protected int m_Attribute = -1;

  /** The split point. */
  protected double m_SplitPoint = Double.NaN;
  
  /** The proportions of training instances going down each branch. */
  // was here just for nominal attributes and missing values
  //protected double[] m_Prop = null;

  /** Class probabilities from the training vals. */
  protected double[] m_ClassProbs = null;


  public FasterTree(FasterTree sucessorLeft, FasterTree sucessorRight, int m_Attribute, double m_SplitPoint, double[] m_ClassProbs) {
    this.sucessorLeft = sucessorLeft;
    this.sucessorRight = sucessorRight;
    this.m_Attribute = m_Attribute;
    this.m_SplitPoint = m_SplitPoint;
    this.m_ClassProbs = m_ClassProbs;
  }

  public FasterTree() {}

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
   * Returns default capabilities of the classifier.
   *
   * @return      the capabilities of this classifier
   */
  @Override
  public Capabilities getCapabilities() {

    Capabilities result = new Capabilities(this);

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
    throw new Exception("FasterTree can be used only by FasterForest " +
            "and FastRfBagger classes, not directly.");
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
   */
  @Override
  public final double[] distributionForInstance(Instance instance) {

    if (m_Attribute == -1) {  // ============================ node is a leaf

      return m_ClassProbs;

    } else { // =============================================== node is not a leaf

      //if (instance.isMissing(m_Attribute)) {  // ---------------- missing value

      //  returnedDist = new double[m_MotherForest.m_Info.numClasses()];
      //  // split instance up
      //  for (int i = 0; i < m_Successors.length; i++) {
      //    double[] help = m_Successors[i].distributionForInstance(instance);
      //    if (help != null) {
      //      for (int j = 0; j < help.length; j++) {
      //        returnedDist[j] += m_Prop[i] * help[j];
      //      }
      //    }
      //  }

      //} else if (m_MotherForest.m_Info
      //        .attribute(m_Attribute).isNominal()) { // ------ nominal

      //  //returnedDist = m_Successors[(int) instance.value(m_Attribute)]
      //  //        .distributionForInstance(instance);

      //  // 0.99: new - binary splits (also) for nominal attributes
      //  if ( instance.value(m_Attribute) == m_SplitPoint ) {
      //    returnedDist = m_Successors[0].distributionForInstance(instance);
      //  } else {
      //    returnedDist = m_Successors[1].distributionForInstance(instance);
      //  }

      //} else { // ------------------------------------------ numeric attributes

      FasterTree node = this;

      while (true) {
        if (instance.value(node.m_Attribute) < node.m_SplitPoint) {
          node = node.sucessorLeft;
        } else {
          node = node.sucessorRight;
        }

        if (node.m_Attribute == -1) {  // node is a leaf
          return node.m_ClassProbs;
        }
      }

    }

  }

  public final double[] distributionForAttributes(double[] instanceAttributes) {


    if (m_Attribute == -1) {  // ============================ node is a leaf

      return m_ClassProbs;

    } else { // =============================================== node is not a leaf

      FasterTree node = this;

      while (true) {
        if (instanceAttributes[m_Attribute] < node.m_SplitPoint) {
          node = node.sucessorLeft;
        } else {
          node = node.sucessorRight;
        }

        if (node.m_Attribute == -1) {  // node is a leaf
          return node.m_ClassProbs;
        }
      }

    }


  }



  /**
   * Computes size of the tree.
   *
   * @return the number of nodes
   */
  public int numNodes() {

    if (m_Attribute == -1) {
      return 1;
    } else {
      int size = 1;
      //for (int i = 0; i < m_Successors.length; i++) {
      //  size += m_Successors[i].numNodes();
      //}

      size += sucessorLeft.numNodes();
      size += sucessorRight.numNodes();

      return size;
    }
  }
  


//  @Override
//  public String getRevision() {
//    return RevisionUtils.extract("$Revision: 0.99.1$");
//  }
//



  /**
   * destroys this tree in the process to free up memory
   */
  public FasterTree toSlimVersion() {
    FasterTree left = sucessorLeft==null ? null : sucessorLeft.toSlimVersion();
    sucessorLeft = null;
    FasterTree right = sucessorRight==null ? null : sucessorRight.toSlimVersion();
    sucessorRight = null;

    FasterTree res = new FasterTree(left, right, m_Attribute, m_SplitPoint, m_ClassProbs);

    return res;
  }



  /**
   * copied from Weka's AbstractClassifier
   *
   * Classifies the given test instance. The instance has to belong to a dataset
   * when it's being classified. Note that a classifier MUST implement either
   * this or distributionForInstance().
   *
   * @param instance the instance to be classified
   * @return the predicted most likely class for the instance or
   *         Utils.missingValue() if no prediction is made
   * @exception Exception if an error occurred during the prediction
   */
  @Override
  public double classifyInstance(Instance instance) throws Exception {

    double[] dist = distributionForInstance(instance);
    if (dist == null) {
      throw new Exception("Null distribution predicted");
    }
    switch (instance.classAttribute().type()) {
      case Attribute.NOMINAL:
        double max = 0;
        int maxIndex = 0;

        for (int i = 0; i < dist.length; i++) {
          if (dist[i] > max) {
            maxIndex = i;
            max = dist[i];
          }
        }
        if (max > 0) {
          return maxIndex;
        } else {
          return Utils.missingValue();
        }
      case Attribute.NUMERIC:
      case Attribute.DATE:
        return dist[0];
      default:
        return Utils.missingValue();
    }
  }

}

