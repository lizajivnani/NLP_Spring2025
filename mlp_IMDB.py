
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

tf.random.set_seed(1234)
np.random.seed(1234)

class MLP_FA(object):
    def __init__(self, size_input, size_hidden1, size_hidden2, size_hidden3, size_output, device=None):
        """
        size_input: int, size of input layer
        size_hidden1: int, size of the 1st hidden layer
        size_hidden2: int, size of the 2nd hidden layer
        size_hidden3: int, size of the 3rd hidden layer (Note: Not used in compute_output in this example)
        size_output: int, size of output layer
        device: str or None, either 'cpu' or 'gpu' or None.
        """
        self.size_input = size_input
        self.size_hidden1 = size_hidden1
        self.size_hidden2 = size_hidden2
        self.size_hidden3 = size_hidden3  # (Currently not used)
        self.size_output = size_output
        self.device = device

        # Initialize weights and biases for first hidden layer
        self.W1 = tf.Variable(tf.random.normal([self.size_input, self.size_hidden1], stddev=0.1))
        self.b1 = tf.Variable(tf.zeros([1, self.size_hidden1]))

        # Initialize weights and biases for second hidden layer
        self.W2 = tf.Variable(tf.random.normal([self.size_hidden1, self.size_hidden2], stddev=0.1))
        self.b2 = tf.Variable(tf.zeros([1, self.size_hidden2]))

        # Initialize weights and biases for output layer
        self.W3 = tf.Variable(tf.random.normal([self.size_hidden2, self.size_output], stddev=0.1))
        self.b3 = tf.Variable(tf.zeros([1, self.size_output]))

        # Create fixed random feedback matrices for feedback alignment:
        # B3: used to propagate the error from the output layer to the second hidden layer.
        # It replaces the use of W3^T. Its shape is (size_output, size_hidden2).
        self.B3 = tf.Variable(tf.random.normal([self.size_output, self.size_hidden2]), trainable=False)

        # B2: used to propagate the error from the second hidden layer to the first hidden layer.
        # Its shape is (size_hidden2, size_hidden1).
        self.B2 = tf.Variable(tf.random.normal([self.size_hidden2, self.size_hidden1]), trainable=False)

        # Define variables to be updated during training
        self.variables = [self.W1, self.W2, self.W3, self.b1, self.b2, self.b3]

    def forward(self, X):
        """
        Forward pass.
        X: Tensor, inputs.
        """
        if self.device is not None:
            with tf.device('gpu:0' if self.device == 'gpu' else 'cpu'):
                self.y = self.compute_output(X)
        else:
            self.y = self.compute_output(X)
        return self.y

    def loss(self, y_pred, y_true):
        """
        Computes the loss between predicted and true outputs.
        y_pred - Tensor of shape (batch_size, size_output)
        y_true - Tensor of shape (batch_size, size_output)
        """
        y_true_tf = tf.cast(y_true, dtype=tf.float32)
        y_pred_tf = tf.cast(y_pred, dtype=tf.float32)
        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        loss_x = cce(y_true_tf, y_pred_tf)
        return loss_x

    def backward(self, X_train, y_train):
        """
        Backward pass using feedback alignment.
        Computes gradients manually using fixed random feedback matrices.
        X_train: Input data (numpy array)
        y_train: One-hot encoded labels (numpy array)
        Returns: List of gradients corresponding to [dW1, dW2, dW3, db1, db2, db3]
        """
        # Cast input to float32 tensor
        X_tf = tf.cast(X_train, tf.float32)

        # --- Forward Pass ---
        # First hidden layer
        h1 = tf.matmul(X_tf, self.W1) + self.b1
        a1 = tf.nn.relu(h1)
        # Second hidden layer
        h2 = tf.matmul(a1, self.W2) + self.b2
        a2 = tf.nn.relu(h2)
        # Output layer (logits)
        logits = tf.matmul(a2, self.W3) + self.b3
        # Softmax predictions
        y_pred = tf.nn.softmax(logits)

        # --- Compute Output Error ---
        # For cross-entropy with softmax, the derivative is (y_pred - y_true)
        delta3 = y_pred - tf.cast(y_train, tf.float32)  # shape: (batch, size_output)
        batch_size = tf.cast(tf.shape(X_tf)[0], tf.float32)

        # --- Gradients for Output Layer ---
        dW3 = tf.matmul(tf.transpose(a2), delta3) / batch_size
        db3 = tf.reduce_mean(delta3, axis=0, keepdims=True)

        # --- Feedback Alignment for Second Hidden Layer ---
        # Instead of delta2 = (delta3 dot W3^T) * ReLU'(h2), use a fixed random matrix B3.
        relu_grad_h2 = tf.cast(h2 > 0, tf.float32)
        # delta3 has shape (batch, size_output) and B3 has shape (size_output, size_hidden2)
        delta2 = tf.matmul(delta3, self.B3) * relu_grad_h2  # shape: (batch, size_hidden2)

        dW2 = tf.matmul(tf.transpose(a1), delta2) / batch_size
        db2 = tf.reduce_mean(delta2, axis=0, keepdims=True)

        # --- Feedback Alignment for First Hidden Layer ---
        # Instead of delta1 = (delta2 dot W2^T) * ReLU'(h1), use a fixed random matrix B2.
        relu_grad_h1 = tf.cast(h1 > 0, tf.float32)
        # delta2 has shape (batch, size_hidden2) and B2 has shape (size_hidden2, size_hidden1)
        delta1 = tf.matmul(delta2, self.B2) * relu_grad_h1  # shape: (batch, size_hidden1)

        dW1 = tf.matmul(tf.transpose(X_tf), delta1) / batch_size
        db1 = tf.reduce_mean(delta1, axis=0, keepdims=True)

        return [dW1, dW2, dW3, db1, db2, db3]

    def compute_output(self, X):
        """
        Custom method to obtain output tensor during the forward pass.
        """
        X_tf = tf.cast(X, dtype=tf.float32)
        h1 = tf.matmul(X_tf, self.W1) + self.b1
        z1 = tf.nn.relu(h1)
        h2 = tf.matmul(z1, self.W2) + self.b2
        z2 = tf.nn.relu(h2)
        output = tf.matmul(z2, self.W3) + self.b3
        return output


# -------------------------------
# Character-Level Tokenizer and Preprocessing Functions
# -------------------------------
def char_level_tokenizer(texts, num_words=None):
    """
    Create and fit a character-level tokenizer.

    Args:
        texts (list of str): List of texts.
        num_words (int or None): Maximum number of tokens to keep.

    Returns:
        tokenizer: A fitted Tokenizer instance.
    """
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_words, char_level=True, lower=True)
    tokenizer.fit_on_texts(texts)
    return tokenizer

def texts_to_bow(tokenizer, texts):
    """
    Convert texts to a bag-of-characters representation.

    Args:
        tokenizer: A fitted character-level Tokenizer.
        texts (list of str): List of texts.

    Returns:
        Numpy array representing the binary bag-of-characters for each text.
    """
    # texts_to_matrix with mode 'binary' produces a fixed-length binary vector per text.
    matrix = tokenizer.texts_to_matrix(texts, mode='binary')
    return matrix

def one_hot_encode(labels, num_classes=2):
    """
    Convert numeric labels to one-hot encoded vectors.
    """
    return np.eye(num_classes)[labels]

# -------------------------------
# Load and Prepare the IMDB Dataset
# -------------------------------
print("Loading IMDB dataset...")
# Load the IMDB reviews dataset with the 'as_supervised' flag so that we get (text, label) pairs.
(ds_train, ds_test), ds_info = tfds.load('imdb_reviews',
                                           split=['train', 'test'],
                                           as_supervised=True,
                                           with_info=True)

# Convert training dataset to lists.
train_texts = []
train_labels = []
for text, label in tfds.as_numpy(ds_train):
    # Decode byte strings to utf-8 strings.
    train_texts.append(text.decode('utf-8'))
    train_labels.append(label)
train_labels = np.array(train_labels)

# Create a validation set from the training data (20% for validation).
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.2, random_state=42)

# Convert test dataset to lists.
test_texts = []
test_labels = []
for text, label in tfds.as_numpy(ds_test):
    test_texts.append(text.decode('utf-8'))
    test_labels.append(label)
test_labels = np.array(test_labels)

print(f"Train samples: {len(train_texts)}, Validation samples: {len(val_texts)}, Test samples: {len(test_texts)}")

# -------------------------------
# Preprocessing: Tokenization and Vectorization
# -------------------------------
# Build the character-level tokenizer on the training texts.
tokenizer = char_level_tokenizer(train_texts)
print("Tokenizer vocabulary size:", len(tokenizer.word_index) + 1)

# Convert texts to bag-of-characters representation.
X_train = texts_to_bow(tokenizer, train_texts)
X_val   = texts_to_bow(tokenizer, val_texts)
X_test  = texts_to_bow(tokenizer, test_texts)

# Convert labels to one-hot encoding.
y_train = one_hot_encode(train_labels)
y_val   = one_hot_encode(val_labels)
y_test  = one_hot_encode(test_labels)

# -------------------------------
# Model Setup
# -------------------------------
# The input size is determined by the dimension of the bag-of-characters vector.
size_input = X_train.shape[1]
# Set hidden layer sizes as desired.
size_hidden1 = 128
size_hidden2 = 64
size_hidden3 = 32  # Placeholder (not used in the forward pass)
size_output  = 2

# Instantiate the MLP model.
model = MLP_FA(size_input, size_hidden1, size_hidden2, size_hidden3, size_output, device=None)

# Define the optimizer.
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# -------------------------------
# Training Parameters and Loop
# -------------------------------
batch_size = 128
epochs = 10
num_batches = int(np.ceil(X_train.shape[0] / batch_size))

print("\nStarting training...\n")
for epoch in range(epochs):
    # Shuffle training data at the start of each epoch.
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]

    epoch_loss = 0
    for i in range(num_batches):
        start = i * batch_size
        end = min((i+1) * batch_size, X_train.shape[0])
        X_batch = X_train[start:end]
        y_batch = y_train[start:end]

        # Compute gradients and update weights.
        # with tf.GradientTape() as tape:
        #     predictions = model.forward(X_batch)
        #     loss_value = model.loss(predictions, y_batch)
        # grads = tape.gradient(loss_value, model.variables)
        predictions = model.forward(X_batch)
        loss_value = model.loss(predictions, y_batch)
        grads = model.backward(X_batch, y_batch)
        optimizer.apply_gradients(zip(grads, model.variables))
        epoch_loss += loss_value.numpy() * (end - start)

    epoch_loss /= X_train.shape[0]

    # Evaluate on validation set.
    val_logits = model.forward(X_val)
    val_loss = model.loss(val_logits, y_val).numpy()
    val_preds = np.argmax(val_logits.numpy(), axis=1)
    true_val = np.argmax(y_val, axis=1)
    accuracy = np.mean(val_preds == true_val)
    precision = precision_score(true_val, val_preds)
    recall = recall_score(true_val, val_preds)

    print(f"Epoch {epoch+1:02d} | Training Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | "
          f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

# -------------------------------
# Final Evaluation on Test Set
# -------------------------------
print("\nEvaluating on test set...")
test_logits = model.forward(X_test)
test_loss = model.loss(test_logits, y_test).numpy()
test_preds = np.argmax(test_logits.numpy(), axis=1)
true_test = np.argmax(y_test, axis=1)
test_accuracy = np.mean(test_preds == true_test)
test_precision = precision_score(true_test, test_preds)
test_recall = recall_score(true_test, test_preds)

print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f} | "
      f"Test Precision: {test_precision:.4f} | Test Recall: {test_recall:.4f}")