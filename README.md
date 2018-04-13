An android application that facilitates farmers, scientists and botanists to detect the type of plant or crops, detect pests and any other kind of diseases in them.
The app sends the image of the plant to the server where it is analysed using CNN classifier model.
Once detected, the disease and its solutions are displayed to the user. Also the closest pesticides selling centers are suggested.

PHASE 1. Collection of DataSet
PHASE 2. Training
PHASE 3. Android Application
PHASE 4. Testing

PHASE 1. Collection of DataSets
Publicly available datasets​: We collect different datasets each for plant type, weeds, disease, pests etc that are publicly available.
Resizing images for efficient storage and prediction.

Datasets:
Horticulture crops taken :
Corn
Peach
Disease categories:
1 healthy for each plant crop
1 diseases for Peach
2 diseases for Corn

PHASE 2: Training
Vectorize each image of dataset when loaded.
Train a CNN (YOLO architecture) on different categories of datasets using keras with tensorflow backend.
Save the weights

PHASE 3: Android Application

PHASE 4: Testing on a leaf image having disease taken from internet.
