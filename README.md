WhirDetective Machine Learning Project
======================================

objectives and context
-----------------------------

Industrial machinery relies on rotating components such as bearings. Over time these bearings wear out and develop defects (inner‑ring faults, outer‑ring faults or rolling element damage), which can lead to catastrophic failures if not detected early. **WhirDetective** is an open‑source Python project that implements an end‑to‑end workflow for bearing fault diagnosis using vibration, current, speed and temperature signals. The aim of this document is to teach you how WhirDetective works, why each component is necessary and to understand the pipeline from scratch.

This document is organised as follows:

1.  **Python foundations** – variables, control flow, functions, classes and modules.
2.  **Numerical computing with NumPy** – arrays, vectorised operations and basic statistics.
3.  **Signal features** – definitions and formulae for statistical and frequency‑domain features used in WhirDetective.
4.  **Segmentation with sliding windows** – converting long time‑series signals into supervised learning samples.
5.  **Machine‑learning concepts** – classification problems, loss functions, data splitting and balanced sampling.
6.  **Neural networks & 1‑D convolution** – high‑level intuition and 1‑D convolutional neural networks (CNNs).
7.  **PyTorch essentials** – tensors, model definition, optimisers and training loops.
8.  **Exploring the WhirDetective codebase** – data pipeline, feature projection, neural network architecture and training workflow.
9.  **Evaluation and calibration** – classification metrics, expected calibration error (ECE), temperature scaling and abstention.
10. **Putting it all together** – how to run WhirDetective and suggestions for further experiments.

Each section introduces concepts from scratch, illustrates them with simplified examples, and then links back to the corresponding module in the WhirDetective repository. When referring to code we show simplified excerpts; you are encouraged to read the full implementation in the repository for deeper understanding.

1 Python foundations
--------------------

Python is a general‑purpose programming language that emphasises readability and concise syntax. WhirDetective is written in Python ≥ 3.11, so this section teaches dedicated to fundamental language features.

### 1.1 Basic data types and variables

*   **Integers and floats:** `int` represents whole numbers and `float` represents real numbers. For example:
    ```python
    count = 10        # an integer
    temperature = 25.8  # a float
    ```
*   **Strings:** text enclosed in single or double quotes. Strings can be concatenated with `+`:
    ```python
    device = "Accelerometer"
    message = "Sensor: " + device
    ```
*   **Booleans:** `True` or `False`. They often result from comparisons, e.g. `is_faulty = vibration > threshold`.

### 1.2 Collections

*   **Lists:** ordered collections that can hold items of any type and are mutable (elements can be changed). Example: `channels = ["accel", "temp", "current"]`.
*   **Tuples:** ordered, immutable sequences often used to group heterogeneous values. Example: `window_config = (1024, 512)`.
*   **Dictionaries:** key–value maps, written using curly braces. Example:
    ```python
    channel_signals = {
        "accel": accel_array,
        "temp": temp_array
    }
    ```
*   **Sets:** unordered collections of unique elements. Useful when you need to ensure uniqueness.

### 1.3 Control flow

Python uses indentation (spaces) to denote blocks of code.

*   **Conditionals:** `if`, `elif` and `else` allow branching.
    ```python
    if sensor == "accel":
        process_acceleration()
    elif sensor == "temp":
        process_temperature()
    else:
        print("Unknown sensor")
    ```
*   **Loops:** `for` loops iterate over sequences, and `while` loops repeat until a condition becomes false. Example:
    ```python
    for sample in samples:
        features = extract_features(sample)
        feature_list.append(features)
    ```
*   **Comprehensions:** concise expressions to build lists, dictionaries or sets in one line. Example:
    ```python
    squared = [x * x for x in range(5)]  # [0, 1, 4, 9, 16]
    ```

### 1.4 Functions and classes

*   **Functions:** reusable blocks defined with `def` that may take arguments and return a value. Example:
    ```python
    def rms(signal: np.ndarray) -> float:
        return np.sqrt(np.mean(signal ** 2))
    ```
*   **Classes:** blueprint for objects that bundle data and behaviour. WhirDetective defines many classes (e.g. `BaselineBearingCNN`, `SensorSetProjector`) as `class MyClass:` with an `__init__` method to initialise attributes. Example:
    ```python
    class SensorSetProjector:
        def __init__(self, policies: dict[str, ProjectionPolicy]):
            self.policies = policies
        def project(self, channel_signals: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
            # returns feature vector and presence mask
            pass
    ```

### 1.5 Modules and packages

Files ending in `.py` are Python modules; directories containing `__init__.py` form packages. To reuse code from another module you use the `import` statement. For example:

```python
import numpy as np
from scipy.io import loadmat  # to load .mat files
```

The WhirDetective codebase is organised as a package named `whirdetective` under the `src` directory. Throughout this document we will refer to modules such as `whirdetective.ml.models` or `whirdetective.data.pipeline`.

2 Numerical computing with NumPy
--------------------------------

Machine‑learning models operate on numerical arrays, and in Python the **NumPy** library provides fast array operations. WhirDetective uses NumPy to process vibration signals and compute statistical features.

### 2.1 Creating arrays

*   **From Python lists:**
    ```python
    import numpy as np
    accel = np.array([0.1, 0.3, -0.2, ...])
    ```
*   **Zeros and ones:** `np.zeros(shape)` returns an array of zeros; `np.ones(shape)` returns ones.
*   **Range of numbers:** `np.arange(start, stop, step)` and `np.linspace(start, stop, num)`.

Arrays have a `shape` attribute describing their dimensions (rows, columns). In vibration analysis each channel signal is a one‑dimensional array of sample values.

### 2.2 Vectorised operations

NumPy performs element‑wise operations without explicit Python loops. For example:

```python
mean_val = np.mean(accel)        # average value
std_val  = np.std(accel)         # standard deviation
rms_val  = np.sqrt(np.mean(accel ** 2))  # root mean square
```

These operations are used in feature extraction (Section 3).

### 2.3 Indexing and slicing

You can access parts of an array using indices. For instance, `signal[0:1024]` returns the first 1024 samples. Negative indices access elements from the end; `signal[-1]` returns the last element.

### 2.4 Broadcasting

NumPy **broadcasting** allows operations between arrays of different shapes by implicitly expanding one array. For example, subtracting the mean from every column in a 2‑D array can be written without loops.

```python
matrix = np.array([[1, 2], [3, 4], [5, 6]])
col_mean = matrix.mean(axis=0)
centered = matrix - col_mean  # subtracts mean from each column
```

Understanding broadcasting is useful when computing per‑channel features across windows.

3 Signal features
-----------------

Statistical and spectral features summarise raw vibration or current signals into fixed‑length vectors that capture important characteristics. The **SensorSetProjector** class in WhirDetective uses the following features by default: mean, standard deviation, root mean square (RMS), peak‑to‑peak (maximum minus minimum), absolute maximum, skewness, kurtosis, crest factor, impulse factor, spectral centroid norm, spectral entropy and spectral flatness[raw.githubusercontent.com](https://raw.githubusercontent.com/MehranKianiHub/WhirDetective/main/src/whirdetective/ml/sensor_projection.py#:~:text=def%20_as_valid_series%28raw_values%3A%20npt,finite%20values%22%29%20return%20series). To understand why these are useful, we briefly define each feature.

### 3.1 Time‑domain features

*   **Mean:** the average value of the signal. It indicates the DC component or offset.
*   **Standard deviation (std):** measures spread around the mean. A larger standard deviation indicates more variability.
*   **Root mean square (RMS):** square root of the mean of squared samples. RMS reflects the effective energy of a vibration signal and is widely used in condition monitoring[mathworks.com](https://www.mathworks.com/help/predmaint/ug/signal-features.html#:~:text=Statistical%20Features). It is calculated as
    $$
    \mathrm{RMS}(x) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2}.
    $$
*   **Peak‑to‑peak:** difference between the maximum and minimum value of a signal. It captures the range of fluctuations.
*   **Absolute maximum:** largest absolute value, i.e.  $\max_i |x_i|$ . Large peaks may indicate impulsive faults.
*   **Skewness:** measure of asymmetry of the probability distribution. Positive skew means the right tail is longer; negative skew means the left tail is longer.
*   **Kurtosis:** measure of “tailedness” or outlier propensity. A high kurtosis implies heavy tails and frequent extreme values. Kurtosis is particularly sensitive to impulsive events in machinery and is thus useful for early fault detection[mathworks.com](https://www.mathworks.com/help/predmaint/ug/signal-features.html#:~:text=Statistical%20Features).
*   **Crest factor:** ratio of the absolute maximum to the RMS. High crest factors can indicate early bearing defects[mathworks.com](https://www.mathworks.com/help/predmaint/ug/signal-features.html#:~:text=Statistical%20Features). Formally:
    $$
    \mathrm{Crest\ Factor}(x) = \frac{\max_i |x_i|}{\mathrm{RMS}(x)}.
    $$
*   **Impulse factor:** ratio of the absolute maximum to the mean of absolute values. It highlights sudden impulses and is defined as
    $$
    \mathrm{Impulse\ Factor}(x) = \frac{\max_i |x_i|}{\frac{1}{N}\sum_{i=1}^N |x_i|}.
    $$

These features summarise how large, variable and asymmetric the signal is, and they are simple to compute using NumPy. In practice, you would compute each feature for each sensor channel in every sliding window (see Section 4) and stack them into a feature vector.

### 3.2 Frequency‑domain features

WhirDetective also computes features derived from the signal’s spectrum. To obtain the spectrum, the signal is first centred by subtracting its mean, then the discrete Fourier transform (DFT) is applied using NumPy’s `np.fft.rfft`. From the magnitudes  $|X(f)|$  the following features are computed[raw.githubusercontent.com](https://raw.githubusercontent.com/MehranKianiHub/WhirDetective/main/src/whirdetective/ml/sensor_projection.py#:~:text=def%20_as_valid_series%28raw_values%3A%20npt,finite%20values%22%29%20return%20series):

*   **Spectral centroid norm:** weighted average frequency  $\sum f |X(f)| / \sum |X(f)|$ . It measures the “centre of mass” of the spectrum.
*   **Spectral entropy:** Shannon entropy of the normalised magnitudes  $p_k = |X(f_k)|/\sum_k |X(f_k)|$ . It quantifies how spread out the energy is; higher entropy means the spectrum is more uniform.
*   **Spectral flatness:** geometric mean divided by the arithmetic mean of the magnitudes. A flatness near 1 indicates a noise‑like, broad spectrum, whereas values near 0 indicate tonal signals.

These spectral features complement time‑domain statistics by capturing frequency content. Spectral entropy and flatness are widely used in signal processing to distinguish periodic from noisy signals[raw.githubusercontent.com](https://raw.githubusercontent.com/MehranKianiHub/WhirDetective/main/src/whirdetective/ml/sensor_projection.py#:~:text=def%20_as_valid_series%28raw_values%3A%20npt,finite%20values%22%29%20return%20series).

4 Segmentation with sliding windows
-----------------------------------

Raw sensor recordings are long time‑series signals. Machine‑learning models require fixed‑length inputs, so WhirDetective splits each channel into overlapping windows using the **sliding window** technique. In this method a window of width  $W$  samples “slides” along the signal, advancing by a step size  $S$  each time. Each window (a segment of the signal) becomes an independent training sample. The window width is also referred to as the lag or history length[machinelearningmastery.com](https://machinelearningmastery.com/time-series-forecasting-supervised-learning/#:~:text=Sliding%20Window%20For%20Time%20Series,Data).

The sliding window technique converts time‑series forecasting into supervised learning by pairing previous time steps (window) with the next value or label[machinelearningmastery.com](https://machinelearningmastery.com/time-series-forecasting-supervised-learning/#:~:text=Sliding%20Window%20For%20Time%20Series,Data). In WhirDetective the label for each window is the fault class associated with the run (e.g. healthy, inner‑ring fault, outer‑ring fault). Overlapping windows mean adjacent samples share data; this increases the number of training examples and improves the model’s ability to detect transient events.

The diagram below illustrates how a signal is segmented into overlapping windows. Each coloured box represents a window of fixed length. In practice, if a signal has length  $N$  samples and the window width is  $W$  with a step size  $S$ , then the number of windows is approximately  $\lfloor (N - W)/S \rfloor + 1$ .

![](1.png)

In code, sliding window segmentation can be implemented with array slicing. For example:

```python
def sliding_windows(signal: np.ndarray, window_size: int, step_size: int) -> list[np.ndarray]:
    windows = []
    for start in range(0, len(signal) - window_size + 1, step_size):
        window = signal[start : start + window_size]
        windows.append(window)
    return windows
```

The **build\_windowed\_canonical\_samples** function in `whirdetective/data/pipeline.py` applies this logic to all available channels, computes features for each window using `SensorSetProjector`, and returns a list of `CanonicalTrainingSample` objects that contain the features, presence mask and metadata[raw.githubusercontent.com](https://raw.githubusercontent.com/MehranKianiHub/WhirDetective/main/src/whirdetective/data/pipeline.py#:~:text=def%20build_windowed_canonical_samples%28%20,channel_signals%20must%20not%20be%20empty).

5 Machine‑learning concepts
---------------------------

Before delving into neural networks, we need to understand some high‑level machine‑learning concepts relevant to WhirDetective.

### 5.1 Classification tasks

A _classification_ problem involves assigning an input to one of a finite number of categories (classes). In WhirDetective the classes correspond to bearing conditions, e.g. healthy, inner‑ring fault, outer‑ring fault or combined faults. Given a feature vector  $\mathbf{x}$  extracted from a window, the model outputs probabilities  $p_c$  for each class  $c$ , and the predicted class is the one with the highest probability.

### 5.2 Loss functions: cross‑entropy

During training, neural networks adjust their parameters to minimise a **loss function** that quantifies prediction errors. The standard loss for classification is the _cross‑entropy_. For binary classification with labels  $y_i \in \{0,1\}$  and predicted probabilities  $p_i$ , the binary cross‑entropy is given by[geeksforgeeks.org](https://www.geeksforgeeks.org/machine-learning/what-is-cross-entropy-loss-function/#:~:text=In%20classification%20problems%2C%20a%20machine,correct%20answers%20in%20classification%20problems)

$$
\mathrm{BCE} = -\frac{1}{N}\sum_{i=1}^N \left[y_i \log(p_i) + (1-y_i) \log(1 - p_i)\right],
$$

where  $N$  is the number of samples. For multiclass classification with  $C$  classes the categorical (softmax) cross‑entropy generalises this idea:

$$
\mathrm{CE} = -\frac{1}{N}\sum_{i=1}^N \sum_{j=1}^C y_{i,j} \log(p_{i,j}),
$$

where  $y_{i,j}$  is 1 if sample  $i$  belongs to class  $j$  and 0 otherwise, and  $p_{i,j}$  is the predicted probability for class  $j$ [geeksforgeeks.org](https://www.geeksforgeeks.org/machine-learning/what-is-cross-entropy-loss-function/#:~:text=Multiclass%20Cross,Entropy%20Loss%20is%20calculated%20as). Cross‑entropy measures how close the predicted probability distribution is to the true distribution; lower values indicate better predictions. It also heavily penalises confident but wrong predictions[geeksforgeeks.org](https://www.geeksforgeeks.org/machine-learning/what-is-cross-entropy-loss-function/#:~:text=In%20classification%20problems%2C%20a%20machine,correct%20answers%20in%20classification%20problems).

### 5.3 Data splitting and leakage avoidance

To evaluate a model’s generalisation, we split data into training, validation and test sets. In condition monitoring, samples may come from the same machine or run, so splitting randomly can leak information (the model might memorise specific patterns rather than learning general features). WhirDetective uses **grouped splitting**: samples are grouped by machine and run, and groups are assigned entirely to train, validation or test sets to prevent leakage[raw.githubusercontent.com](https://raw.githubusercontent.com/MehranKianiHub/WhirDetective/main/src/whirdetective/data/engine.py#:~:text=run_samples%20%3D%20build_windowed_canonical_samples%28%20dataset%3D,len%28run_samples%29%29%20source_files.append%28file_path). The `split_by_group` function in `whirdetective/data/splitting.py` implements this strategy, and the `build_cwru_canonical_dataset` and `build_paderborn_canonical_dataset` functions apply it when constructing datasets[raw.githubusercontent.com](https://raw.githubusercontent.com/MehranKianiHub/WhirDetective/main/src/whirdetective/data/engine.py#:~:text=run_samples%20%3D%20build_windowed_canonical_samples%28%20dataset%3D,len%28run_samples%29%29%20source_files.append%28file_path)[raw.githubusercontent.com](https://raw.githubusercontent.com/MehranKianiHub/WhirDetective/main/src/whirdetective/data/engine.py#:~:text=def%20build_paderborn_canonical_dataset%28%20,rar%20archives%20for%20train%2Fval%2Ftest%20split).

### 5.4 Balanced sampling

Bearing datasets are often imbalanced; for example there may be many more healthy windows than fault windows. The `BaselineTrainer` in WhirDetective optionally uses _balanced sampling_ to ensure that each mini‑batch contains an equal number of samples from each class. This helps the network learn to recognise minority classes. Balanced sampling is implemented via PyTorch’s `WeightedRandomSampler`, which assigns inverse‑frequency weights to each sample during training.

6 Neural networks and 1‑D convolution
-------------------------------------

Neural networks consist of layers of interconnected units (neurons) that transform inputs into outputs via learnable weights. Convolutional neural networks (CNNs) are a specialised type of neural network designed to process data with spatial or temporal structure, such as images or vibration signals.

### 6.1 Why convolution?

Convolutional layers “slide” learnable filters across the input and compute dot products to detect local patterns. Unlike traditional fully connected layers, convolutional layers exploit _local connectivity_ and _weight sharing_. This means that each filter is applied across the entire input to recognise a particular feature regardless of its position. As a result, CNNs learn to autonomously extract features at multiple scales and are translation‑invariant[datacamp.com](https://www.datacamp.com/tutorial/introduction-to-convolutional-neural-networks-cnns#:~:text=,in%20position%2C%20orientation%2C%20scale%2C%20or).

CNNs were inspired by the hierarchical architecture of the human visual cortex; early layers detect simple patterns, deeper layers build more complex features. Local connectivity, translation invariance and multiple feature maps are key characteristics[datacamp.com](https://www.datacamp.com/tutorial/introduction-to-convolutional-neural-networks-cnns#:~:text=,maps%20in%20each%20convolution%20layer). Although CNNs originated in image analysis, the same principles apply to one‑dimensional signals such as vibration or audio.

### 6.2 1‑D convolution

WhirDetective uses a **1‑D convolutional neural network**. Instead of sliding a 2‑D kernel over an image, 1‑D convolution slides a kernel along a sequence. For example, a convolution layer with a kernel size of 5 and stride 2 computes overlapping weighted sums of every five adjacent samples of each channel. This allows the network to learn local temporal patterns, such as periodic vibrations or fault signatures.

Mathematically, given an input sequence  $x$  of length  $N$  and a kernel (filter)  $w$  of length  $K$ , the discrete convolution at position  $t$  is

$$
(x * w)_t = \sum_{k=0}^{K-1} x_{t+k} \cdot w_k.
$$

The kernel parameters  $w_k$  are learned during training. Multiple kernels form a layer with multiple output channels; non‑linearities (e.g. ReLU) and pooling further transform the representations.

### 6.3 WhirDetective’s CNN architecture

The `BaselineBearingCNN` defined in `whirdetective/ml/models.py` is a simple yet effective architecture:

1.  **Input:** feature vectors of shape `(batch_size, num_features, 1)` (the `1` is the time dimension after the projection). Since features are already aggregated across channels, the convolution layers use kernel size 1.
2.  **Three convolutional blocks:** each block consists of a 1‑D convolution (`nn.Conv1d`), batch normalisation (`nn.BatchNorm1d`) and ReLU activation. The number of output channels increases from 32 to 128 across the blocks[raw.githubusercontent.com](https://raw.githubusercontent.com/MehranKianiHub/WhirDetective/main/src/whirdetective/ml/models.py#:~:text=self,Linear%2864%2C%20num_classes%29%2C).
3.  **Adaptive average pooling:** reduces the temporal dimension to 1 by computing the mean across timesteps. This ensures that the network can handle variable‑length inputs.
4.  **Fully connected layer:** a linear classifier mapping the flattened features to class logits.

Despite its simplicity, the network achieves good performance on bearing fault datasets because most of the discriminative information has already been encoded in the time‑frequency features computed by `SensorSetProjector`. Deeper or more complex networks could be substituted if necessary.

7 PyTorch essentials
--------------------

WhirDetective uses PyTorch as its deep‑learning framework. Key concepts include tensors, modules, optimisers and training loops.

### 7.1 Tensors

PyTorch’s core data structure is the **tensor**, a multidimensional array similar to a NumPy array but with automatic differentiation support. To convert NumPy arrays to PyTorch tensors:

```python
import torch
import numpy as np

x_np = np.array([1, 2, 3])
x_torch = torch.tensor(x_np, dtype=torch.float32)
```

Tensors have a `shape` and support operations such as `matmul`, `sum`, `mean` and broadcasting. If a tensor requires gradients (for learning) you set `requires_grad=True`.

### 7.2 Modules and model definition

In PyTorch, models are built by subclassing `torch.nn.Module` and defining layers in `__init__`, then specifying the forward computation in `forward`. For example, the simplified `BaselineBearingCNN` looks like this:

```python
import torch.nn as nn

class BaselineBearingCNN(nn.Module):
    def __init__(self, input_channels: int, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # flatten
        return self.classifier(x)
```

### 7.3 Optimisers and training loop

Training a neural network involves iteratively updating its parameters to minimise the loss. WhirDetective uses the **Adam** optimiser, a popular variant of stochastic gradient descent. A simplified training loop looks like:

```python
model = BaselineBearingCNN(input_channels=num_features, num_classes=num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    for batch_features, batch_labels in dataloader:
        optimizer.zero_grad()
        logits = model(batch_features)
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()
    # optionally evaluate on validation set here
```

The `BaselineTrainer` in `whirdetective/training/trainer.py` wraps this loop, supports balanced sampling, and computes evaluation metrics after each epoch.

8 Exploring the WhirDetective codebase
--------------------------------------

WhirDetective is modular and follows a clear separation of concerns. This section outlines the main components, showing how they interact.

### 8.1 Data adapters and loaders

The raw bearing datasets (CWRU and Paderborn) are stored in `.mat` files (CWRU) or `.rar` archives (Paderborn). Adapter functions in `whirdetective/data/adapters` handle dataset‑specific details:

*   `list_cwru_mat_files` and `load_cwru_channels`: enumerate `.mat` files in the CWRU directory and load acceleration, current, speed and temperature channels.
*   `infer_cwru_label_from_path`: derive the fault label from the file path.
*   `list_paderborn_archives`, `list_paderborn_mat_entries` and `load_paderborn_channels_from_mat_payload`: perform similar functions for the Paderborn dataset.

These adapters return raw NumPy arrays per channel and a label. They do not compute features or windows; that is handled by the pipeline.

### 8.2 Canonical training samples and features

`CanonicalTrainingSample` (defined in `whirdetective/data/contracts.py`) encapsulates a single sample with the following fields[raw.githubusercontent.com](https://raw.githubusercontent.com/MehranKianiHub/WhirDetective/main/src/whirdetective/data/contracts.py#:~:text=%40dataclass%28frozen%3DTrue%2C%20slots%3DTrue%29%20class%20CanonicalTrainingSample%3A%20%22%22%22Fixed,agnostic%20sample%20contract%20for%20training%2Fevaluation):

*   `dataset`: string identifying the dataset (e.g. "cwru" or "paderborn").
*   `machine_id` and `run_id`: identifiers to group samples by machine/run for leakage‑free splitting.
*   `label`: enumerated `BearingFaultLabel`.
*   `features`: NumPy array containing concatenated features for all sensor channels.
*   `presence_mask`: binary mask indicating which sensors were available in the original sample.
*   `metadata`: dictionary for arbitrary information (e.g. sample index, timestamp).

The **SensorSetProjector** (in `whirdetective/ml/sensor_projection.py`) converts variable‑length, multi‑channel signals into a fixed‑length feature vector and presence mask. It uses a configurable `ProjectionPolicy` for each sensor type (e.g. acceleration, temperature, current). When you call `projector.project(channel_signals)`, it iterates through channels, computes the features from Section 3 and concatenates them. If a channel is missing, the corresponding portion of the presence mask is set to 0 and the feature values are filled with zeros[raw.githubusercontent.com](https://raw.githubusercontent.com/MehranKianiHub/WhirDetective/main/src/whirdetective/ml/sensor_projection.py#:~:text=_DEFAULT_FEATURE_NAMES%3A%20tuple,)

### 8.3 Dataset engine

The `whirdetective/data/engine.py` module orchestrates dataset construction. The key functions are `build_cwru_canonical_dataset` and `build_paderborn_canonical_dataset`. They perform the following steps[raw.githubusercontent.com](https://raw.githubusercontent.com/MehranKianiHub/WhirDetective/main/src/whirdetective/data/engine.py#:~:text=run_samples%20%3D%20build_windowed_canonical_samples%28%20dataset%3D,len%28run_samples%29%29%20source_files.append%28file_path)[raw.githubusercontent.com](https://raw.githubusercontent.com/MehranKianiHub/WhirDetective/main/src/whirdetective/data/engine.py#:~:text=def%20build_paderborn_canonical_dataset%28%20,rar%20archives%20for%20train%2Fval%2Ftest%20split):

1.  **List and select files/archives** from the dataset root directory. You may limit the number of files/archives for quick experiments.
2.  **Load channels and labels** using the adapter functions.
3.  **Segment signals into windows** using `build_windowed_canonical_samples` (Section 4).
4.  **Project signals to features** using a `SensorSetProjector`.
5.  **Assign group identifiers** of the form `machine_id:run_id` to each sample.
6.  **Split into train/validation/test** using `split_by_group` (Section 5.3) with configurable ratios and constraints (e.g. requiring all labels to appear in each split).
7.  **Return a `BuiltCanonicalDataset`** containing the samples, group IDs, split information and dataset fingerprint for reproducibility.

### 8.4 Model training workflow

The training pipeline is orchestrated in `whirdetective/training/workflow.py`. The main function, `run_baseline_training_workflow`, performs the following high‑level steps[raw.githubusercontent.com](https://raw.githubusercontent.com/MehranKianiHub/WhirDetective/main/src/whirdetective/training/workflow.py#:~:text=,0%2C%201)[raw.githubusercontent.com](https://raw.githubusercontent.com/MehranKianiHub/WhirDetective/main/src/whirdetective/training/workflow.py#:~:text=dim%3D1%29,numpy):

1.  **Build the canonical dataset** (Section 8.3).
2.  **Create PyTorch datasets and data loaders** for the training, validation and test splits.
3.  **Instantiate the model** (`BaselineBearingCNN`) and a `BaselineTrainer` with hyperparameters such as epochs, batch size, learning rate and weight decay.
4.  **Train the model** using the trainer. The training loop records loss and accuracy per epoch.
5.  **Calibrate the model**: After training, temperature scaling is applied to the logits to improve calibration (Section 9.2). Calibration is performed on the validation set.
6.  **Select an abstention threshold**: Using the calibrated probabilities, an optional threshold is selected to abstain on low‑confidence predictions (reducing errors at the cost of coverage).
7.  **Evaluate metrics**: The workflow computes classification, calibration and abstention metrics on the test set using functions from `whirdetective/evaluation/metrics.py`[raw.githubusercontent.com](https://raw.githubusercontent.com/MehranKianiHub/WhirDetective/main/src/whirdetective/evaluation/metrics.py#:~:text=def%20compute_classification_metrics%28%20labels%3A%20npt,num_classes%20must%20be%20%3E%201)[raw.githubusercontent.com](https://raw.githubusercontent.com/MehranKianiHub/WhirDetective/main/src/whirdetective/evaluation/metrics.py#:~:text=def%20compute_calibration_metrics%28%20probabilities%3A%20npt,num_bins%20must%20be%20%3E%200).
8.  **Create a model card** summarising the model’s performance and metadata.

This workflow can be run from the command line via `whirdetective-runner` as defined in `pyproject.toml`. For example:

```bash
whirdetective-runner \
    --dataset_root /path/to/cwru \
    --model_output_dir /tmp/whirdetective_output \
    --window_size 2048 \
    --step_size 512 \
    --epochs 20
```

9 Evaluation and calibration
----------------------------

### 9.1 Classification metrics

The `ClassificationMetrics` dataclass in `whirdetective/evaluation/metrics.py` computes the following:

*   **Confusion matrix:** a square matrix where entry  $M_{i,j}$  counts predictions of class  $j$  for true class  $i$ . From the confusion matrix one can compute per‑class recall (true positive rate) and precision.
*   **Accuracy:** the proportion of correctly classified samples:  $\text{accuracy} = \frac{1}{N} \sum_{i=1}^N \mathbb{1}[y_i = \hat{y}_i]$ , where  $\hat{y}_i$  is the predicted class for sample  $i$ [raw.githubusercontent.com](https://raw.githubusercontent.com/MehranKianiHub/WhirDetective/main/src/whirdetective/evaluation/metrics.py#:~:text=def%20compute_classification_metrics%28%20labels%3A%20npt,num_classes%20must%20be%20%3E%201).
*   **Balanced accuracy:** average of per‑class recall. This metric is useful when classes are imbalanced.

### 9.2 Calibration and Expected Calibration Error (ECE)

A classifier is _calibrated_ when its predicted probabilities reflect true likelihoods. For example, among all predictions with confidence 0.8, roughly 80 % should be correct[iclr-blogposts.github.io](https://iclr-blogposts.github.io/2025/blog/calibration/#:~:text=Calibration%20makes%20sure%20that%20a,many%20applications%20across%20various%20domains). Poor calibration can lead to over‑ or under‑confident predictions, which is problematic in safety‑critical applications.

The **Expected Calibration Error (ECE)** is a common metric for measuring calibration. It partitions predictions into bins of equal width (e.g. 10 bins over the range 0 to 1) and computes, for each bin, the difference between the average confidence and the average accuracy. ECE is the weighted average of these differences[iclr-blogposts.github.io](https://iclr-blogposts.github.io/2025/blog/calibration/#:~:text=Evaluating%20Calibration%20,ECE). Formally, if  $B_m$  is the set of indices whose maximum predicted probability falls into bin  $m$ , then

$$
\mathrm{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \bigl| \mathrm{acc}(B_m) - \mathrm{conf}(B_m) \bigr|,
$$

where  $\mathrm{acc}(B_m)$  is the proportion of correct predictions in bin  $m$  and  $\mathrm{conf}(B_m)$  is the average predicted probability[iclr-blogposts.github.io](https://iclr-blogposts.github.io/2025/blog/calibration/#:~:text=Evaluating%20Calibration%20,ECE).

To improve calibration, WhirDetective applies **temperature scaling**: a single scalar parameter  $T > 1$  rescales the logits  $z$  before the softmax as  $\hat{p} = \mathrm{softmax}(z / T)$ . The temperature is optimised on the validation set to minimise the negative log‑likelihood. After calibration the probabilities are better aligned with empirical frequencies.

### 9.3 Abstention and coverage

Sometimes it is preferable for a model to abstain rather than risk an incorrect prediction. An **abstention threshold**  $\tau$  is chosen such that if the maximum predicted probability is below  $\tau$ , the model withholds the prediction. The **coverage** is the proportion of samples with confidence ≥  $\tau$ . **Selective accuracy** is the accuracy computed only over covered samples[raw.githubusercontent.com](https://raw.githubusercontent.com/MehranKianiHub/WhirDetective/main/src/whirdetective/evaluation/metrics.py#:~:text=def%20compute_calibration_metrics%28%20probabilities%3A%20npt,num_bins%20must%20be%20%3E%200). By tuning  $\tau$ , one can trade off coverage against reliability.

10 Putting it all together
--------------------------

This section outlines how you can replicate WhirDetective’s pipeline end‑to‑end.

1.  **Install dependencies:** ensure Python ≥ 3.11 and install `numpy`, `torch` and other dependencies listed in `pyproject.toml` (optional extras include `matplotlib`, `scipy`, `tqdm`). A virtual environment is recommended.
2.  **Download datasets:** obtain the Case Western Reserve University (CWRU) bearing dataset and/or the Paderborn dataset. Organise them in directories expected by WhirDetective (e.g. each `.mat` file in CWRU corresponds to one run).
3.  **Configure window size and step size:** typical values are 1024–4096 samples for vibration signals; smaller windows capture shorter events but produce more samples.
4.  **Create a `SensorSetProjector`:** decide which channels to use (e.g. acceleration only vs. acceleration + current + temperature) and which features. The default policies compute all features from Section 3[raw.githubusercontent.com](https://raw.githubusercontent.com/MehranKianiHub/WhirDetective/main/src/whirdetective/ml/sensor_projection.py#:~:text=_DEFAULT_FEATURE_NAMES%3A%20tuple,).
5.  **Build the canonical dataset:** call `build_cwru_canonical_dataset` or `build_paderborn_canonical_dataset` with the chosen window/step sizes and projector. Inspect the returned `BuiltCanonicalDataset` to see how many samples each split contains.
6.  **Train the model:** instantiate a `BaselineBearingCNN` with input channels equal to the number of computed features and output classes equal to the number of fault types. Use the `BaselineTrainer` to train for several epochs, enabling balanced sampling if necessary.
7.  **Calibrate and evaluate:** after training, optimise the temperature on the validation set, select an abstention threshold and compute classification, calibration and abstention metrics on the test set. Visualise the reliability diagram to understand calibration.
8.  **Experiment and improve:** try different feature sets (e.g. remove spectral features or add wavelet features), adjust the network architecture (deeper CNN, residual connections), or apply data augmentation (adding noise or random shifts). Compare performance across datasets and analyse misclassified examples.

11 Further study
----------------

WhirDetective provides a solid foundation for bearing fault diagnosis, but there is much more to explore:

*   **Model interpretability:** investigate which features or time segments contribute most to predictions; methods like Grad‑CAM can highlight important regions in CNNs.
*   **Prognostics:** the `whirdetective/prognostics` module hints at remaining useful life estimation and degradation trends; exploring these requires understanding regression models and health indicators.
*   **Safety and deployment:** the `whirdetective/safety` and `whirdetective/export` modules support runtime monitoring and export to portable formats (e.g. ONNX, TorchScript) for deployment on embedded devices.
*   **Modern deep architectures:** beyond the baseline CNN, recurrent neural networks (RNNs), transformers or graph neural networks can be applied to time‑series data. Experimentation can reveal whether more sophisticated models improve fault detection.

Conclusion
----------

This document provided a step‑by‑step introduction to Python, NumPy, signal features, sliding windows, neural networks, PyTorch and the WhirDetective codebase. Citations were included to emphasise definitions and formulas. By following the examples and exploring the code, you should be able to understand the entire WhirDetective pipeline. The combination of careful feature engineering and a simple convolutional neural network demonstrates how domain knowledge and machine‑learning techniques can be integrated to solve a real‑world industrial problem.

The document includes citations from credible sources throughout, ensuring that definitions and formulas are well-supported.

<p align="center">
  <strong>⭐ Star us on GitHub if you find this project useful! ⭐</strong>
</p>

<p align="center">
  Made with ❤️ 
</p>

---