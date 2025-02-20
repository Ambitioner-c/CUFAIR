### **CUFAIR**: <u>C</u>omment-based <u>U</u>nified use<u>F</u>ul <u>A</u>nswer <u>I</u>dentification <u>F</u>ramework
___

This is our implementation for the paper:

`Yidong Chai`, Fulai Cui, Shuo Yu, Yuanchun Jiang, Haoxin Liu, Yezheng Liu. Identifying Useful Answers on Community-Based Question Answering Platforms: A Novel Unified Answer Comment-Based Approach. Preprint at https://ssrn.com/abstract=5026989, 2024.

For more information, feel free to [Fulai Cui](mailto:cuifulai@mail.hfut.edu.cn).
### Environment Settings
- Python 3.12.4
- Required libraries: `numpy`, `pandas`, `pytorch`, `scikit-learn`, etc.

### Example to Run the Codes
- Run the following command to train the `CUFAIR` model on the <u>StackExchange</u> dataset.
```
python OM.py --model_name CUFAIR --data_name meta.stackoverflow.com --device cuda:1 --batch_size 4 --lr 2e-5 --alpha 0.8 --margin 5 --seed 2024
```

### Model Implementation Details
| Component                     | Description                                  | Details                                             |
|-------------------------------|----------------------------------------------|-----------------------------------------------------|
| **Hyperparameters**           | λ                                            | 0.8                                                 |
|                               | γ                                            | 5                                                   |
|                               | learning rate                                | 2e-5                                                |
|                               | batch size                                   | 8                                                   |
|                               | random seed                                  | 2024                                                |
| **Embedding Layer**           | Max Sequence Length                          | 256                                                 |
|                               | BERT Embedding Dimension                     | 768                                                 |
|                               | Dimensionality Reduction Embedding Dimension | 108                                                 |
| **Argument Quality**          | Depth Feature Dimension                      | 7                                                   |
|                               | Readability Feature Dimension                | 11                                                  |
|                               | Objectivity Feature Dimension                | 4                                                   |
|                               | Timeliness Feature Dimension                 | 4                                                   |
|                               | Accuracy Feature Dimension                   | 8                                                   |
|                               | Structure Feature Dimension                  | 10                                                  |
|                               | Relevancy Feature Dimension                  | 20                                                  |
|  **Source Credibility**       | Max Comments Number                          | 5                                                   |
|                               | Num Attention Heads                          | 12                                                  |
|                               | CE-LSTM                                      | CE-LSTM(num_layers=1, num_attention_heads=12)       |
|                               | Community Support Representation Layer       | Linear(in_features=108, out_features=64, bias=True) |
|                               | Community Support Score Layer                | Linear(in_features=64, out_features=1, bias=True)   |
| **Output**                    | Usefulness Layer                             | Linear(in_features=128, out_features=1, bias=True)  |

### Dataset
- The dataset used in this paper is the <u>meta.stackoverflow.com</u> dataset, which can be downloaded from [here](https://archive.org/details/stackexchange).

Last Update Date: February 20, 2025