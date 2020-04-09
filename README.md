### Siamese Network For One Shot Face Verification

> I have created a 5 layer ( 3 Conv + 2 FC) neural network and trained it on Labeled Faces In Wild Dataset with randomly generated labels to generate 128 x 1 embedding.

**Using The Embeddings**

```python3
import cv2
from net import Net64

face = cv2.imread("./face.jpg)
other_face = cv2.imread("./other_face.jpg")

net  = Net64(base=face,weights_path="./path/to/weights.h5")


print (net(face))
print (net(other_face))
```

```bash
>> 0.0
>> 122.7
```

**Embedding Layers**

...

**Scoring Layers**

**1. RMSE**
    