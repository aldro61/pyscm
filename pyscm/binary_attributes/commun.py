import numpy as np

from .base import BinaryAttributeListMixin

class DefaultBinaryAttributeList(BinaryAttributeListMixin):

    def __init__(self, binary_attributes):
        self.binary_attributes = binary_attributes

    def __len__(self):
        return len(self.binary_attributes)

    def __getitem__(self, item_idx):
        return self.binary_attributes[item_idx]

    def classify(self, X):
        attribute_classifications = np.zeros((X.shape[0], len(self)), dtype=np.uint8)
        for i, ba in enumerate(self.binary_attributes):
            attribute_classifications[:, i] = ba.classify(X)
        return attribute_classifications


