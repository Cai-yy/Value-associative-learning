import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from util import *
import os
import datetime


# %%
# ablation type, 0 for no ablation, 1 for no diversity loss,
# 2 for additive context input, 3 for decision loss in all connections
ablatype = 3

# %% Define input structure
gotype = [1, 2]
nogotype = [3, 4]
nC = 4  # Number of cues
nfre = 4  # Number of frequencies in each chord
input_size = nC * nfre
cueM = np.zeros([nC, input_size], np.double)
for ccc in range(nC):
    cueM[ccc, ccc::nfre] = 1


# Function, generate input and target, using variable initial delay
def genXY(cueM, nTini, nTcue, nTdelay, nTdecision, nTgap, ampInput):
    nTtrial = nTini + nTcue + nTdelay + nTdecision + nTgap
    # Generate inputs
    x = torch.zeros([cueM.shape[1], nTtrial, 4])
    target = torch.zeros([nTtrial, 4])
    target[nTini + nTcue + nTdelay : nTini + nTcue + nTdelay + nTdecision, 0:2] = 1
    target[nTini + nTcue + nTdelay : nTini + nTcue + nTdelay + nTdecision, 2:4] = -1
    for icc in range(4):
        for tt in range(nTtrial):
            if tt < nTini + nTcue and tt >= nTini:
                x[:, tt, icc] = torch.tensor(cueM[icc, :]).to(torch.float32)
            else:
                x[:, tt, icc] = torch.zeros(1, input_size)
    x = x * ampInput  # Set the amplitude of input
    return x, target, nTtrial


# %% Define network
# Set netork size
output_size = 1
output_size_ran = 10
tauC = 5  # Temporal paramter of node
tauWN = 3  # Temporal parameter of noise
ampWN = 0.1  # Noise amplitude
nN1 = 100  # neuron number of first module
nN2 = 100  # neuron number of second module
nN3 = 100  # neuron number of third module
fracinput = 1  # fraction of neurons that receive input from cue
fracinter = 0.3  # fraction of neurons receive input from other module
nN = nN1 + nN2 + nN3
hidden_size = nN1 + nN2 + nN3
g = 1.5
ampInput = 1

# Set trial structure
nTcue = 20
nTdelay = 2
nTdecision = 10
nTgap = 10


# input
# x = torch.randn(1, input_size)  # initialization of input
h = torch.zeros(1, nN).to(torch.float32)  # initialization of hidden node


# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu') # Because there is frequent transformation of device, its not worth it to put in GPU

# Outpath
outname = (
    "threemodule_gating_sparse_"
    + datetime.datetime.now().strftime("%m%d")
    + "_abla"
    + str(ablatype)
)
outdir = "./output/" + outname
os.makedirs(outdir, exist_ok=True)

nRepeat = 14
nEpoch = 2000
batchsize = 16
for rrr in range(nRepeat):
    outputr = os.path.join(outdir, f"repeat_{rrr}")
    os.makedirs(outputr, exist_ok=True)

    #  Training
    # Innitialize network
    if ablatype in [0,1]:
        model = threeModuleRNN_sparse_context_gating(
            input_size,
            nN1,
            nN2,
            nN3,
            output_size,
            output_size_ran,
            tauC,
            tauWN,
            ampWN,
            g,
            fracinput,
            fracinter,
            2,
        )
    elif ablatype==2:
        model = threeModuleRNN_sp_context_additive(
            input_size,
            nN1,
            nN2,
            nN3,
            output_size,
            output_size_ran,
            tauC,
            tauWN,
            ampWN,
            g,
            fracinput,
            fracinter,
            2,
        )
    elif ablatype==3:
        model = threeModuleRNN_sp_context_alldeloss(
            input_size,
            nN1,
            nN2,
            nN3,
            output_size,
            output_size_ran,
            tauC,
            tauWN,
            ampWN,
            g,
            fracinput,
            fracinter,
            2,
        )

    # optimizer
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )

    model = model.to(device)
    for epoch in tqdm(range(nEpoch)):
        optimizer.zero_grad()

        #  Define trial structure
        nTini = np.random.randint(10, 30)  # Initial delay in each trial
        # Batched input
        x, target, nTtrial = genXY(
            cueM, nTini, nTcue, nTdelay, nTdecision, nTgap, ampInput
        )
        xb = torch.zeros([batchsize, cueM.shape[1], nTtrial])  # Input with noise
        tb = torch.zeros([batchsize, nTtrial])  # Target
        h = torch.zeros([batchsize, hidden_size])  # Initial state
        cuelist = np.random.randint(0, 4, size=batchsize)
        for bbb in range(batchsize):
            curcid = cuelist[bbb]
            # Add a bit noise to each time input
            xb[bbb, :, :] = x[:, :, curcid] + torch.randn(x[:, :, curcid].size()) / 2
            tb[bbb, :] = target[:, curcid]
            h[bbb, :] = torch.randn(1, hidden_size).to(
                torch.float32
            )  # initialization of hidden node

        # to device
        xb = xb.to(device)
        h = h.to(device)
        # Active trial
        HH, output, randreadout = model(xb, h, [1, 0])

        # calculate loss
        # Decision loss
        loss_ad = 0
        for t in range(nTtrial):
            if t < nTini + nTcue + nTdelay | t >= nTini + nTcue + nTdelay + nTdecision:
                loss_ad += nn.MSELoss()(output[:, t], tb[:, t]) * 0
            else:
                loss_ad += nn.MSELoss()(output[:, t], tb[:, t])
        # loss=nn.MSELoss()(output[:, nTcue+nTdelay:nTcue+nTdelay+nTdecision], tb[:, nTcue+nTdelay:nTcue+nTdelay+nTdecision])
        loss_ad = loss_ad / nTdecision
        # Entropy loss, this loss term encourage the A cortex to output various behavior
        # loss_va = rand_readout_mixloss(HH,cuelist,[nTini,nTini+nTcue+nTdecision+nTdelay])
        loss_va = anti_cluster_covariance_loss(randreadout, [10, nTtrial])
        # Rand readout loss
        loss_randr_a = (
            nn.MSELoss()(randreadout, torch.randn(randreadout.shape) * 1e-2) * 0.2
        )
        # L1 norm
        l1_penalty = model.interconnectL1()

        # Decision loss only effects part of the gradient, as well as the variability loss
        for losstype in [0]:
            if losstype == 0:  # Decision loss
                param = model.weight_decision_loss()
                grads = torch.autograd.grad(loss_ad, param, retain_graph=True)
            elif losstype == 1:
                param = model.weight_vari_loss()
                grads = torch.autograd.grad(loss_va, param, retain_graph=True)
            else:
                param = model.weight_l1_loss()
                grads = torch.autograd.grad(l1_penalty, param, retain_graph=True)

            for p, grr in zip(param, grads):
                if p.grad is None:
                    p.grad = grr
                else:
                    p.grad += grr

        # update
        optimizer.step()

        # Passive trial
        xb = torch.zeros([batchsize, cueM.shape[1], nTtrial])  # Input with noise
        tb = torch.zeros([batchsize, nTtrial])  # Target is set to zero
        h = torch.zeros([batchsize, hidden_size])  # Initial state
        cuelist = np.random.randint(0, 4, size=batchsize)
        for bbb in range(batchsize):
            curcid = cuelist[bbb]
            # Add a bit noise to each time input
            xb[bbb, :, :] = x[:, :, curcid] + torch.randn(x[:, :, curcid].size()) / 2
            h[bbb, :] = torch.randn(1, hidden_size).to(
                torch.float32
            )  # initialization of hidden node
        xb = xb.to(device)
        h = h.to(device)

        # Forward
        HHp, output, randreadout_p = model(xb, h, [0, 1])
        # Loss
        loss_pd = nn.MSELoss()(output, tb)
        loss_vp = anti_cluster_covariance_loss(randreadout_p, [10, nTtrial])
        l1_penalty = model.interconnectL1()

        # Sum loss
        if ablatype == 1:
            losstypes = [0]
        else:
            losstypes = [0, 1]
        for losstype in losstypes:
            if losstype == 0:  # Decision loss
                param = model.weight_decision_loss()
                grads = torch.autograd.grad(loss_pd, param, retain_graph=True)
            elif losstype == 1:
                param = model.weight_vari_loss()
                grads = torch.autograd.grad(loss_vp, param, retain_graph=True)
            else:
                param = model.weight_l1_loss()
                grads = torch.autograd.grad(l1_penalty, param, retain_graph=True)

            for p, grr in zip(param, grads):
                if p.grad is None:
                    p.grad = grr
                else:
                    p.grad += grr

        # Update
        optimizer.step()

        if epoch % 10 == 0:
            tqdm.write(
                f"R {rrr} Epoch [{epoch}/{nEpoch}], Loss ad: {loss_ad.item()}, ar: {loss_va.item()}, pd: {loss_pd.item()}, pr: {loss_vp.item()}, l1:{l1_penalty.item()} "
            )
            if epoch % 100 == 0:
                # Save weight
                model.saveWeight(outputr, epoch)
                tosave = {}

                # Show spontaneous activity
                with torch.no_grad():
                    xb = torch.zeros([1, cueM.shape[1], 500]).to(device)  # Zero input
                    h = torch.randn([1, hidden_size]).to(device)  # Initial state
                    Fr_sp, y_sp, _ = model(xb, h, [0, 0])
                    # plt.imshow(
                    #     Fr_sp.detach().cpu().squeeze().numpy(),
                    #     cmap="hot",
                    #     interpolation="nearest",
                    # )
                    # plt.savefig(os.path.join(outdir, f"SpontAct_e{epoch}.png"))
                    # plt.colorbar()
                    # plt.close()

                # Show test output
                with torch.no_grad():
                    x, _, nTtrial = genXY(
                        cueM, 20, nTcue, nTdelay, nTdecision, nTgap, ampInput
                    )  # with initial delay 20 as input
                    nRp = 20  # Repeat for 20 times
                    Resp = np.zeros([nN, nTtrial, 4, 2, nRp])
                    for irr in range(nRp):
                        for curc in range(4):
                            for iaa, av in enumerate([[1, 0], [0, 1]]):
                                h0 = (
                                    torch.randn(1, hidden_size)
                                    .to(torch.float32)
                                    .to(device)
                                )
                                xx = (
                                    (
                                        x[:, :, curc]
                                        + torch.randn(x[:, :, curc].size()) / 2
                                    )
                                    .unsqueeze(0)
                                    .to(device)
                                )
                                hh, output, _ = model(xx, h0, av)
                                # Collect
                                Resp[:, :, curc, iaa, irr] = (
                                    hh.squeeze().detach().cpu().numpy()
                                )
                                if irr == 0:
                                    plt.plot(
                                        np.squeeze(
                                            output.squeeze().detach().cpu().numpy()
                                        ),
                                        label=f"cue {curc}-{iaa}",
                                    )

                    plt.legend()
                    plt.savefig(os.path.join(outputr, f"testCue_e{epoch}.png"))
                    plt.close()
                    tosave["Resp"] = Resp
                    tosave["Fr_sp"] = Fr_sp.detach().cpu().numpy()
                    # Save the trained model
                    torch.save(
                        model, os.path.join(outputr, f"trained_model_e{epoch}.pth")
                    )
                    savemat(os.path.join(outputr, f"simres_e{epoch}.mat"), tosave)
