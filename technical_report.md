# **Technical Report: Semantic Segmentation with Sparse Point Annotations**

## **1. Method**
This project addresses **semantic segmentation with limited annotation data**. We train a **DeepLabV3-ResNet50** model using only **100 randomly sampled point annotations per image** (instead of complete pixel masks) on the **LandCover.AI dataset** for aerial image segmentation.

### **1.1 Partial Focal Loss**
We implement a custom loss function that enables training with sparse annotations:
```
Focal Loss = -α(1-pt)^γ × log(pt)
Partial Loss = Σ(Focal_Loss × Binary_Mask) / Σ(Binary_Mask)
```
- **α = 0.25**, **γ = 2.0**: Focal loss parameters for handling class imbalance
- **Binary_Mask**: Indicates which pixels are labeled (1) or unlabeled (0)
- Computes gradients only at labeled points while leveraging full ground truth

### **1.2 Model Architecture**
- **Base**: DeepLabV3-ResNet50 with Atrous Spatial Pyramid Pooling (ASPP)
- **Classes**: 5 (background, building, woodland, water, road)
- **Dataset**: LandCover.AI (7,470 training images, 512×512 resolution)

---

## **2. Experiments & Hypotheses**

| **Experiment** | **Hypothesis** |
|----------------|----------------|
| **Pretrained vs. Non-Pretrained** | Pretrained models will leverage ImageNet transfer learning for better generalization |
| **Learning Rate (0.01 vs 0.001)** | Moderate learning rate (0.001) will provide more stable convergence |
| **Epoch Count (5 vs 10)** | More epochs allow better propagation of information from sparse points |

---

## **3. Results**

### **Experiment 1: Pretrained vs. Non-Pretrained**
| Configuration | Final Loss |
|--------------|------------|
| Pretrained (ImageNet) | **0.0503** ✓ |
| Non-Pretrained | 0.0611 |

**Finding**: Pretrained model achieved **18% lower loss**, demonstrating that ImageNet transfer learning provides beneficial feature representations for aerial imagery segmentation.

### **Experiment 2: Learning Rate Effect**
| Learning Rate | Final Loss |
|--------------|------------|
| 0.01 (High) | 0.0590 |
| 0.001 (Moderate) | **0.0430** ✓ |

**Finding**: Moderate learning rate achieved **27% better loss** with more stable optimization.

### **Experiment 3: Epoch Count**
| Epochs | Final Loss |
|--------|------------|
| 5 | 0.0551 |
| 10 | **0.0471** ✓ |

**Finding**: Doubling epochs resulted in **15% loss reduction**, showing the model benefits from extended training with sparse supervision.

---

## **4. Conclusions**

**Best Configuration:**
- Pretrained: True
- Learning Rate: 0.001
- Epochs: 10+

**Key Insights:**
1. Pretrained models with ImageNet weights provide better performance through transfer learning
2. Learning rate significantly impacts performance (27% improvement with 0.001 vs 0.01)
3. Extended training and sparse annotation (100 points) enable effective segmentation learning

[Detailed experiment report](experiment_report.txt)
