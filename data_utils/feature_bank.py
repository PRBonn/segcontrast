import torch

class ContrastiveFeatureBank:
    def __init__(self, stack_size=512):
        self.stack_size = stack_size
        self.feat_bank_i = None
        self.feat_bank_j = None
    
    def add_batch(self, batch_i, batch_j):
        batch_i = batch_i
        batch_j = batch_j

        assert_msg = 'batch_i and batch_j should have the same shape'
        assert batch_i.shape == batch_j.shape, assert_msg

        assert_msg = 'the new batched features should be smaller or equal to the feature bank stack size'
        assert batch_i.shape[0] <= self.stack_size, assert_msg

        if self.feat_bank_i is None or self.feat_bank_j is None:
            self.feat_bank_i = batch_i
            self.feat_bank_j = batch_j
            return

        # the new feat is added to the beginning
        # the current feature bank is moved to remove the n last items
        slide_i = self.stack_size - batch_i.shape[0]
        self.feat_bank_i = torch.cat((batch_i, self.feat_bank_i[:slide_i]))

        slide_j = self.stack_size - batch_j.shape[0]
        self.feat_bank_j = torch.cat((batch_j, self.feat_bank_j[:slide_j]))

    def get_feat_bank(self):
        return self.feat_bank_i, self.feat_bank_j