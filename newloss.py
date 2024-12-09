import torch
from torch import nn
from regularizer import L1


class OverusePenaltyLoss(nn.Module):
    def __init__(self, temp=0.001, q_reg=0.0, d_reg=0.0, lambda_I=0.001, lambda_T=0.001, T=1000):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.q_regularizer = L1(q_reg, T)
        self.d_regularizer = L1(d_reg, T)
        self.temp = temp
        self.lambda_I = lambda_I
        self.lambda_T = lambda_T

    def overuse_penalty(self, sparse_representations):
        N, V = sparse_representations.size()
        print(sparse_representations)
        print(sparse_representations.mean(dim=0))
        avg_activation = sparse_representations.mean(dim=0)  # Average activation per vocabulary term
        penalty = (avg_activation ** 3).sum() / avg_activation.sum()
        print(penalty)
        return penalty

    def forward(self, sparse_texts, sparse_imgs, dense_texts, dense_imgs):
        # Calculate sparse similarity scores
        sparse_i2t_scores = sparse_imgs @ sparse_texts.t()
        sparse_t2i_scores = sparse_i2t_scores.t()

        # Calculate dense similarity scores for soft labels
        with torch.no_grad():
            scores_dense_i2t = dense_imgs @ dense_texts.t()
            prob_dense_i2t = torch.softmax(scores_dense_i2t / self.temp, dim=1)
            prob_dense_t2i = torch.softmax(scores_dense_i2t.t() / self.temp, dim=1)

        # Cross-entropy loss
        loss = (self.ce(sparse_i2t_scores, prob_dense_i2t) + self.ce(sparse_t2i_scores, prob_dense_t2i)) / 2

        # Regularization terms
        reg = (self.q_regularizer(sparse_texts) + self.d_regularizer(sparse_imgs)) / 2

        # Overuse penalties
        overuse_penalty_text = self.overuse_penalty(sparse_texts)
        overuse_penalty_img = self.overuse_penalty(sparse_imgs)

        # Total loss
        total_loss = loss + reg + self.lambda_T * overuse_penalty_text + self.lambda_I * overuse_penalty_img

        # Step the regularizers
        self.q_regularizer.step()
        self.d_regularizer.step()

        return total_loss, reg
