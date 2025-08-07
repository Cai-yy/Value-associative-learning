import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.io import savemat
import torch.nn.functional


class threeModuleRNN_sparse_context_gating(nn.Module):
    # The struture of this network is module1 (nN1) is directly connected to input, module 2 (nN2) receives input from module 1,
    #  and module3 (nN3) receives input from nN2 and connects to the output
    def __init__(
        self,
        input_size,
        nN1,
        nN2,
        nN3,
        output_size,
        output_size_ran,
        tau,
        tauWN,
        ampWN,
        g,
        fracinput,
        fracinter,
        contexttype,
    ):
        super(threeModuleRNN_sparse_context_gating, self).__init__()

        self.nN1 = nN1
        self.nN2 = nN2
        self.nN3 = nN3
        self.nN = nN1 + nN2 + nN3
        self.input_size = input_size
        self.output_size = output_size
        self.output_size_ran = output_size_ran
        self.g = g

        # Weight, from input to hidden
        self.W_ih = nn.Parameter(
            torch.randn(nN1, input_size) * self.g / np.sqrt(input_size + nN1)
        )  # (hidden_size, input_size)
        with torch.no_grad():
            self.W_ih[round(nN1 * (fracinput)) :, :] = 0  # Keep only a part of input

        # Weight, recurrent connection in hidden nodes
        self.W_H_11 = nn.Parameter(
            torch.randn(nN1, nN1) * self.g / np.sqrt(nN1)
        )  # (hidden_size, hidden_size)
        self.W_H_22 = nn.Parameter(torch.randn(nN2, nN2) * self.g / np.sqrt(nN2 + nN1))
        self.W_H_33 = nn.Parameter(torch.randn(nN3, nN3) * self.g / np.sqrt(nN3 + nN2))

        # Weight, inter mdoules
        idN12_1 = torch.randperm(nN1)
        self.idN12_1 = idN12_1[0 : round(nN1 * fracinter)]
        idN12_2 = torch.randperm(nN2)
        self.idN12_2 = idN12_2[0 : round(nN2 * fracinter)]
        idN23_2 = torch.randperm(nN2)
        self.idN23_2 = idN23_2[0 : round(nN2 * fracinter)]
        idN23_3 = torch.randperm(nN3)
        self.idN23_3 = idN23_3[0 : round(nN3 * fracinter)]
        self.W_H_12 = nn.Parameter(
            torch.randn(len(self.idN12_2), len(self.idN12_1))
            * self.g
            / np.sqrt(nN2 + nN1)
        )
        self.W_H_23 = nn.Parameter(
            torch.randn(len(self.idN23_3), len(self.idN23_2))
            * self.g
            / np.sqrt(nN3 + nN2)
        )

        # Weight, from hidden nodes to output
        self.W_ho = nn.Parameter(
            torch.randn(output_size, nN3) * self.g / np.sqrt(nN3)
        )  # (output_size, hidden_size)

        # Weight, context
        self.W_Hc2 = nn.Parameter(torch.randn(nN2, contexttype) * self.g / np.sqrt(nN2))
        self.W_Hc1 = nn.Parameter(torch.randn(nN1, contexttype) * self.g / np.sqrt(nN1))

        # Fix the recurrent of A, and the input and the output weight
        self.W_ih.requires_grad = False
        # self.W_H_22.requires_grad = False
        # self.W_H_11.requires_grad = False
        # self.W_H_33.requires_grad = False

        self.tau = tau
        self.tauWN = tauWN
        self.ampWN = ampWN

    def forward(self, x, h, context):  # Multi timestep forward propogation
        # x: input, batch size * channel number * time points
        # h: initial hidden state
        # context: [1,0] for active, [0,1] for passive

        batchsize, _, seq_len = x.size()
        h_t = h  # batch size * (nN1+nN2)
        H = torch.zeros(batchsize, self.nN, seq_len)
        y = torch.zeros(batchsize, seq_len)
        yr = torch.zeros(batchsize, self.output_size_ran, 3, seq_len)
        # Add a noise term
        iWN = np.sqrt(self.tauWN) * torch.randn(self.nN, seq_len)
        inputWN = iWN
        for t in range(1, seq_len):
            inputWN[:, t] = iWN[:, t] + (inputWN[:, t - 1] - iWN[:, t]) * np.exp(
                -1 / self.tauWN
            )
        inputWN = self.ampWN * inputWN
        inputWN = inputWN.to(x.device)
        # print(inputWN.shape)

        # A randomized output weight matrix
        W_ho_ran_2 = nn.Parameter(
            torch.randn(self.output_size_ran, self.nN2) * self.g / np.sqrt(self.nN2)
        ).to(x.device)

        W_ho_ran_3 = nn.Parameter(
            torch.randn(self.output_size_ran, self.nN3) * self.g / np.sqrt(self.nN3)
        ).to(x.device)

        W_ho_ran_1 = nn.Parameter(
            torch.randn(self.output_size_ran, self.nN1) * self.g / np.sqrt(self.nN1)
        ).to(x.device)

        # Iterate overtime
        for t in range(seq_len):
            x_t = x[:, :, t]

            # Module 1
            h_t_M1 = h_t[:, : self.nN1]
            WNm1 = inputWN[: self.nN1, t]
            h_t_M1_new = (
                h_t_M1
                + (
                    -h_t_M1
                    + torch.matmul(
                        torch.tanh(h_t_M1), self.W_H_11.t()
                    )  # Recurrent connection
                    + torch.matmul(x_t, self.W_ih.t())  # Input
                    + WNm1  # Noise
                )
                / self.tau
            )

            # Module 2
            h_t_M2 = h_t[:, self.nN1 : self.nN1 + self.nN2]
            WNm2 = inputWN[self.nN1 : self.nN1 + self.nN2, t]
            fromM1 = torch.zeros([batchsize, self.nN1]).to(x.device)
            fromM1[:, self.idN12_2] = torch.matmul(
                torch.tanh(h_t_M1[:, self.idN12_1]), self.W_H_12.t()
            )  # From module 1
            contextinput = torch.matmul(
                context[0] * torch.ones([batchsize, 1]).to(x.device),
                self.W_Hc2[:, 0].t().unsqueeze(0),
            )  # Context 1 input
            +torch.matmul(
                context[1] * torch.ones([batchsize, 1]).to(x.device),
                self.W_Hc2[:, 1].t().unsqueeze(0),
            )  # Context 2 input

            h_t_M2_new = (
                h_t_M2
                + (
                    -h_t_M2
                    + torch.matmul(
                        torch.tanh(h_t_M2), self.W_H_22.t()
                    )  # Recurrent connection
                    + torch.mul(
                        fromM1, contextinput
                    )  # From module 1 modulated by context
                    + WNm2  # Noise
                )
                / self.tau
            )

            # Module 3
            h_t_M3 = h_t[:, self.nN1 + self.nN2 :]
            WNm3 = inputWN[self.nN1 + self.nN2 :, t]
            fromM2 = torch.zeros([batchsize, self.nN1]).to(x.device)
            fromM2[:, self.idN23_3] = torch.matmul(
                torch.tanh(h_t_M2[:, self.idN23_2]), self.W_H_23.t()
            )  # From module 2
            h_t_M3_new = (
                h_t_M3
                + (
                    -h_t_M3
                    + torch.matmul(
                        torch.tanh(h_t_M3), self.W_H_33.t()
                    )  # Recurrent connection
                    + fromM2
                    + WNm3  # Noise
                )
                / self.tau
            )

            # Cat
            h_t = torch.cat([h_t_M1_new, h_t_M2_new, h_t_M3_new], dim=1)
            H[:, :, t] = torch.tanh(h_t)

            # Output decision
            y[:, t] = torch.tanh(
                torch.matmul(torch.tanh(h_t_M3), self.W_ho.t())
            ).squeeze(-1)

            # Random readout
            yr[:, :, 1, t] = torch.tanh(
                torch.matmul(torch.tanh(h_t_M2), W_ho_ran_2.t())
            ).squeeze(-1)
            yr[:, :, 2, t] = torch.tanh(
                torch.matmul(torch.tanh(h_t_M3), W_ho_ran_3.t())
            ).squeeze(-1)
            yr[:, :, 0, t] = torch.tanh(
                torch.matmul(torch.tanh(h_t_M1), W_ho_ran_1.t())
            ).squeeze(-1)

        return H, y, yr

    def saveWeight(self, savedir, epoch):
        weight_dict = {}
        weight_dict["H_ih"] = self.W_ih.detach().cpu().numpy()
        weight_dict["H_11"] = self.W_H_11.detach().cpu().numpy()
        weight_dict["H_12"] = self.W_H_12.detach().cpu().numpy()
        weight_dict["H_22"] = self.W_H_22.detach().cpu().numpy()
        weight_dict["H_ho"] = self.W_ho.detach().cpu().numpy()
        weight_dict["H_33"] = self.W_H_33.detach().cpu().numpy()
        weight_dict["H_23"] = self.W_H_23.detach().cpu().numpy()
        weight_dict["W_Hc2"] = self.W_Hc2.detach().cpu().numpy()
        weight_dict["W_Hc1"] = self.W_Hc1.detach().cpu().numpy()
        weight_dict["idN12_1"] = self.idN12_1.detach().cpu().numpy()
        weight_dict["idN12_2"] = self.idN12_2.detach().cpu().numpy()
        weight_dict["idN23_2"] = self.idN23_2.detach().cpu().numpy()
        weight_dict["idN23_3"] = self.idN23_3.detach().cpu().numpy()

        savemat(os.path.join(savedir, f"Model_weights_e{epoch}.mat"), weight_dict)

    def interconnectL1(self):
        l1loss = torch.mean(abs(self.W_H_12)) + torch.mean(abs(self.W_H_23))
        return l1loss * 10

    def weight_decision_loss(self):
        return [
            self.W_H_12,
            self.W_Hc2,
            self.W_H_23,
            self.W_ho,
            self.W_H_22,
            self.W_H_33,
        ]

    def weight_vari_loss(self):
        return [
            self.W_H_11,
            self.W_H_12,
            self.W_Hc2,
            self.W_H_23,
            self.W_H_22,
            self.W_H_33,
        ]

    def weight_l1_loss(self):
        return [self.W_H_12, self.W_H_23]


# In this class, context is additively added
class threeModuleRNN_sp_context_additive(threeModuleRNN_sparse_context_gating):
    # Override the forward function in this class
    def forward(self, x, h, context):  # Multi timestep forward propogation
        # x: input, batch size * channel number * time points
        # h: initial hidden state
        # context: [1,0] for active, [0,1] for passive

        batchsize, _, seq_len = x.size()
        h_t = h  # batch size * (nN1+nN2)
        H = torch.zeros(batchsize, self.nN, seq_len)
        y = torch.zeros(batchsize, seq_len)
        yr = torch.zeros(batchsize, self.output_size_ran, 3, seq_len)
        # Add a noise term
        iWN = np.sqrt(self.tauWN) * torch.randn(self.nN, seq_len)
        inputWN = iWN
        for t in range(1, seq_len):
            inputWN[:, t] = iWN[:, t] + (inputWN[:, t - 1] - iWN[:, t]) * np.exp(
                -1 / self.tauWN
            )
        inputWN = self.ampWN * inputWN
        inputWN = inputWN.to(x.device)
        # print(inputWN.shape)

        # A randomized output weight matrix
        W_ho_ran_2 = nn.Parameter(
            torch.randn(self.output_size_ran, self.nN2) * self.g / np.sqrt(self.nN2)
        ).to(x.device)

        W_ho_ran_3 = nn.Parameter(
            torch.randn(self.output_size_ran, self.nN3) * self.g / np.sqrt(self.nN3)
        ).to(x.device)

        W_ho_ran_1 = nn.Parameter(
            torch.randn(self.output_size_ran, self.nN1) * self.g / np.sqrt(self.nN1)
        ).to(x.device)

        # Iterate overtime
        for t in range(seq_len):
            x_t = x[:, :, t]

            # Module 1
            h_t_M1 = h_t[:, : self.nN1]
            WNm1 = inputWN[: self.nN1, t]
            h_t_M1_new = (
                h_t_M1
                + (
                    -h_t_M1
                    + torch.matmul(
                        torch.tanh(h_t_M1), self.W_H_11.t()
                    )  # Recurrent connection
                    + torch.matmul(x_t, self.W_ih.t())  # Input
                    + WNm1  # Noise
                )
                / self.tau
            )

            # Module 2
            h_t_M2 = h_t[:, self.nN1 : self.nN1 + self.nN2]
            WNm2 = inputWN[self.nN1 : self.nN1 + self.nN2, t]
            fromM1 = torch.zeros([batchsize, self.nN1]).to(x.device)
            fromM1[:, self.idN12_2] = torch.matmul(
                torch.tanh(h_t_M1[:, self.idN12_1]), self.W_H_12.t()
            )  # From module 1
            contextinput = torch.matmul(
                context[0] * torch.ones([batchsize, 1]).to(x.device),
                self.W_Hc2[:, 0].t().unsqueeze(0),
            )  # Context 1 input
            +torch.matmul(
                context[1] * torch.ones([batchsize, 1]).to(x.device),
                self.W_Hc2[:, 1].t().unsqueeze(0),
            )  # Context 2 input

            h_t_M2_new = (
                h_t_M2
                + (
                    -h_t_M2
                    + torch.matmul(
                        torch.tanh(h_t_M2), self.W_H_22.t()
                    )  # Recurrent connection
                    + torch.add(
                        fromM1, contextinput
                    )  # From module 1 modulated by context
                    + WNm2  # Noise
                )
                / self.tau
            )

            # Module 3
            h_t_M3 = h_t[:, self.nN1 + self.nN2 :]
            WNm3 = inputWN[self.nN1 + self.nN2 :, t]
            fromM2 = torch.zeros([batchsize, self.nN1]).to(x.device)
            fromM2[:, self.idN23_3] = torch.matmul(
                torch.tanh(h_t_M2[:, self.idN23_2]), self.W_H_23.t()
            )  # From module 2
            h_t_M3_new = (
                h_t_M3
                + (
                    -h_t_M3
                    + torch.matmul(
                        torch.tanh(h_t_M3), self.W_H_33.t()
                    )  # Recurrent connection
                    + fromM2
                    + WNm3  # Noise
                )
                / self.tau
            )

            # Cat
            h_t = torch.cat([h_t_M1_new, h_t_M2_new, h_t_M3_new], dim=1)
            H[:, :, t] = torch.tanh(h_t)

            # Output decision
            y[:, t] = torch.tanh(
                torch.matmul(torch.tanh(h_t_M3), self.W_ho.t())
            ).squeeze(-1)

            # Random readout
            yr[:, :, 1, t] = torch.tanh(
                torch.matmul(torch.tanh(h_t_M2), W_ho_ran_2.t())
            ).squeeze(-1)
            yr[:, :, 2, t] = torch.tanh(
                torch.matmul(torch.tanh(h_t_M3), W_ho_ran_3.t())
            ).squeeze(-1)
            yr[:, :, 0, t] = torch.tanh(
                torch.matmul(torch.tanh(h_t_M1), W_ho_ran_1.t())
            ).squeeze(-1)

        return H, y, yr


# In this class, deicision loss is imposed on the inner module connectivity of A
class threeModuleRNN_sp_context_alldeloss(threeModuleRNN_sparse_context_gating):
    def weight_decision_loss(self):
        return [
            self.W_H_11, # This is the difference between this and the original model
            self.W_H_12,
            self.W_Hc2,
            self.W_H_23,
            self.W_ho,
            self.W_H_22,
            self.W_H_33,
        ]


def anti_cluster_covariance_loss(yr, tw, eps=1e-5):
    # x: (batch_size, readoutchannel, readoutmodulenum, time_length)
    loss = 0
    # print(H.shape)
    max_ds = (tw[1] - tw[0]) // 2
    for icc in range(yr.shape[1]):
        for jj in [1, 2]:  # Do not calculate loss on the auditory cortex
            for dsrate in np.logspace(2, np.log2(max_ds), num=4, base=2):
                x = yr[:, icc, jj, tw[0] : tw[1]].squeeze()
                # x = H.reshape(H.shape[0]* H.shape[1],H.shape[2])
                ksize = min(round(dsrate), x.shape[-1])
                # print(ksize)
                x = (x - x.mean(dim=1, keepdim=True)) / x.std(
                    dim=1, keepdim=True
                )  # 去中心
                xx = torch.nn.functional.avg_pool1d(x, kernel_size=ksize, stride=1)
                cov = torch.matmul(xx, xx.T) / (xx.size(1) - 1)  # 协方差矩阵 (B x B)
                # cov = torch.matmul(x, x.T)  # (B, B)
                identity = torch.eye(cov.size(0), device=x.device)
                cov = cov + eps * identity  # 避免奇异
                sign, logabsdet = torch.slogdet(cov)
                log_normalized_det = logabsdet / cov.size(0)  # 归一化 log det
                loss -= log_normalized_det  # maximize log(det) -> minimize negative
    return loss * 1e-3
