from __future__ import absolute_import

from .triplet import TripletLoss, SoftTripletLoss,SimpleCSE
from .crossentropy import CrossEntropyLabelSmooth, SoftEntropy

__all__ = [
    'TripletLoss',
    'CrossEntropyLabelSmooth',
    'SoftTripletLoss',
    'SoftEntropy',
    'SimpleCSE'
]
