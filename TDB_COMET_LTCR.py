import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from models import TSEncoder, ProjectionHead
from tsai.all import GAP1d
from models.losses import contrastive_loss
from models.losses import sample_contrastive_loss, observation_contrastive_loss, patient_contrastive_loss, trial_contrastive_loss
from utils import batch_shuffle_feature_label, trial_shuffle_feature_label, shuffle_feature_label
from utils import MyBatchSampler
import math
import copy
import sklearn


class COMET:
    """The COMET model"""
    
    def __init__(
        self,
        input_dims,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        device='cuda',
        lr=0.001,
        batch_size=128,
        after_iter_callback=None,
        after_epoch_callback=None,
        flag_use_multi_gpu=True
    ):
        """ Initialize a COMET model.
        
        Args:
            input_dims (int): The input dimension. For a uni-variate time series, this should be set to 1.
            output_dims (int): The representation dimension.
            hidden_dims (int): The hidden dimension of the encoder.
            depth (int): The number of hidden residual blocks in the encoder.
            device (str): The gpu used for training and inference.
            lr (float): The learning rate.
            batch_size (int): The batch size of samples.
            after_iter_callback (Union[Callable, NoneType]): A callback function that would be called after each iteration.
            after_epoch_callback (Union[Callable, NoneType]): A callback function that would be called after each epoch.
            flag_use_multi_gpu (bool): A flag to indicate whether using multiple gpus
        """
        
        super().__init__()
        self.device = device
        self.lr = 0.001
        self.batch_size = batch_size

        self.output_dims = output_dims
        self.hidden_dims = hidden_dims

        self.flag_use_multi_gpu = flag_use_multi_gpu
        # gpu_idx_list = [0, 1]
        self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth)
        device = torch.device(device)
        if device == torch.device('cuda') and self.flag_use_multi_gpu:
            # self._net = nn.DataParallel(self._net, device_ids=gpu_idx_list)
            self._net = nn.DataParallel(self._net)
        self._net.to(device)
        # stochastic weight averaging
        # https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
        # self.net = self._net
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)
        self.gap = GAP1d(1)
        self.fc1 = nn.Linear(2*self.output_dims, self.output_dims)
        self.fc2 = nn.Linear(self.output_dims, self.output_dims)
        self.fc3 = nn.Linear(self.output_dims, 2) # 1280
        self.fc1.to(device)
        self.fc2.to(device)
        self.fc3.to(device)

        self.criterion = nn.CrossEntropyLoss()
        # projection head append after encoder
        # self.proj_head = ProjectionHead(input_dims=self.output_dims, output_dims=2, hidden_dims=128).to(self.device)
        
        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback
        
        self.n_epochs = 1
        self.n_iters = 1
    
    def fit(self, X, y, shuffle_function='trial', masks=None, factors=None, n_epochs=None, n_iters=None, verbose=True):
        """ Training the COMET model.
        
        Args:
            X (numpy.ndarray): The training data. It should have a shape of (n_samples, sample_timestamps, features).
            y (numpy.ndarray): The training labels. It should have a shape of (n_samples, 3). The three columns are the label, patient id, and trial id.
            shuffle_function (str): specify the shuffle function.
            masks (list): A list of masking functions applied (str). [Patient, Trial, Sample, Observation].
            factors (list): A list of loss factors. [Patient, Trial, Sample, Observation].
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
            verbose (bool): Whether to print the training loss after each epoch.
            
        Returns:
            epoch_loss_list: a list containing the training losses on each epoch.
            epoch_f1_list: a list containing the linear evaluation on validation f1 score on each epoch.
        """
        assert X.ndim == 3
        assert y.shape[1] == 4
        # Important!!! Shuffle the training set for contrastive learning pretraining. Check details in utils.py.
        X, y = shuffle_feature_label(X, y, shuffle_function=shuffle_function, batch_size=self.batch_size)

        if n_iters is None and n_epochs is None:
            n_iters = 200 if X.size <= 100000 else 600  # default param for n_iters

        # we need patient id for patient-level contrasting and trial id for trial-level contrasting
        train_dataset = TensorDataset(torch.from_numpy(X).to(torch.float), torch.from_numpy(y).to(torch.float))
        if shuffle_function == 'random':
            train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True,
                                      drop_last=False)
        else:
            # Important!!! A customized batch_sampler to shuffle samples before each epoch. Check details in utils.py.
            my_sampler = MyBatchSampler(range(len(train_dataset)), batch_size=min(self.batch_size, len(train_dataset)), drop_last=False)
            train_loader = DataLoader(train_dataset, batch_sampler=my_sampler)
        
        mlp_parameters = []
        mlp_parameters.extend(list(self.fc1.parameters()))
        mlp_parameters.extend(list(self.fc2.parameters()))
        mlp_parameters.extend(list(self.fc3.parameters()))
        # 
        model_parameters = list(self._net.parameters())
        all_parameters = model_parameters + mlp_parameters
        optimizer = torch.optim.AdamW(all_parameters, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50, min_lr=0.0001)
        
        epoch_loss_list, epoch_f1_list = [], []
        
        while True:
            # count by epoch
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            
            cum_loss = 0
            n_epoch_iters = 1
            
            interrupted = False
            for x, y in train_loader:
                # count by iterations
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break

                x = x.to(self.device)
                pid = y[:, 1].to(self.device)  # patient id
                tid = y[:, 2].to(self.device)  # trial id
                sid = y[:, 3].to(self.device)  # sample id

                optimizer.zero_grad()

                # positive pairs construction
                # the fixed random seed guarantee reproducible
                # but does not mean the same mask will generate same result in one running

                if masks is None:
                    masks = ['all_true', 'all_true', 'continuous', 'continuous']

                if factors is None:
                    factors = [0.25, 0.25, 0.25, 0.25]

                if factors[0] != 0:
                    # do augmentation and compute representation
                    patient_out1 = self._net(x, mask=masks[0])
                    patient_out2 = self._net(x, mask=masks[0])

                    # loss calculation
                    patient_loss = contrastive_loss(
                        patient_out1,
                        patient_out2,
                        patient_contrastive_loss,
                        id=pid,
                        hierarchical=False,
                        factor=factors[0],
                    )
                else:
                    patient_loss = 0

                if factors[1] != 0:
                    trial_out1 = self._net(x, mask=masks[1])
                    trial_out2 = self._net(x, mask=masks[1])

                    trial_loss = contrastive_loss(
                        trial_out1,
                        trial_out2,
                        trial_contrastive_loss,
                        id=tid,
                        hierarchical=False,
                        factor=factors[1],
                    )
                else:
                    trial_loss = 0

                if factors[2] != 0:
                    sample_out1 = self._net(x, mask=masks[2])
                    sample_out2 = self._net(x, mask=masks[2])

                    sample_loss = contrastive_loss(
                        sample_out1,
                        sample_out2,
                        sample_contrastive_loss,
                        hierarchical=True,
                        factor=factors[2],
                    )
                else:
                    sample_loss = 0

                if factors[3] != 0:
                    observation_out1 = self._net(x, mask=masks[3])
                    observation_out2 = self._net(x, mask=masks[3])

                    observation_loss = contrastive_loss(
                        observation_out1,
                        observation_out2,
                        observation_contrastive_loss,
                        hierarchical=True,
                        factor=factors[3],
                    )
                else:
                    observation_loss = 0

                # trend and cycle 
                cycle_loss, trend_loss = 0.0, 0.0
                mapping_dict1, mapping_dict2 = {}, {} # CLS
                mapping_dict1[1] = torch.tensor(0).view(1)
                mapping_dict1[-1] = torch.tensor(1).view(1)
                mapping_dict2[1] = torch.tensor(0).view(1)
                mapping_dict2[2] = torch.tensor(1).view(1)
                mapping_dict2[-1] = torch.tensor(2).view(1)
                mapping_dict2[-2] = torch.tensor(3).view(1)
                if(1):
                    # self._net.repr_dropout.eval()
                    sample_relative = self._net(x, mask='all_true') # torch.Size([256, 256, 320])
                    sample_relative = sample_relative.permute(0, 2, 1) # torch.Size([256, 320, 256])
                    # sample_relative_gap = self.gap(sample_relative) # torch.Size([256, 320])
                    # self._net.repr_dropout.train()   
                    unique_tids, counts = torch.unique(tid, return_counts=True)
                    # selected_tids = unique_tids[counts >= 2] #                  
                    # selected_tids = unique_tids[counts >= 9] #            
                    selected_tids = unique_tids[counts >= 19] #  
                    if selected_tids.shape[0] > 1:
                        
                        cycle426_cls, cycle426_cls_labels, cycle426_contrast = [], [], [[],[],[]]
                        cycle640_cls, cycle640_cls_labels, cycle640_contrast = [], [], [[],[]]

                        for selected_tid in selected_tids:
                            #   
                            mask = (tid == selected_tid).flatten()
                            selected_samples = sample_relative[mask]
                            absolute_positions = sid[mask]
                            sorted_indices = torch.argsort(absolute_positions)
                            # 
                            selected_samples = selected_samples[sorted_indices]
                            # print("here:",selected_samples.shape,absolute_positions.shape)

                            # deal with cycle1 = 640
                            for i in range(9):
                                # selected_samples[i][:,0:173],[i+3][:,42:215],[i+6][:,84:] 
                                tt = (self.fc1(torch.cat([self.gap(selected_samples[i]),self.gap(selected_samples[i+10])], dim=0).squeeze()))
                                tt = (self.fc2(tt))
                                tt = self.fc3(tt)
                                cycle640_cls.append(tt.view(1,2)) # 
                                cycle640_cls_labels.append(mapping_dict1[1])
                                tt = (self.fc1(torch.cat([self.gap(selected_samples[i+10]),self.gap(selected_samples[i])], dim=0).squeeze()))
                                tt = (self.fc2(tt))
                                tt = self.fc3(tt)
                                cycle640_cls.append(tt.view(1,2)) # 
                                cycle640_cls_labels.append(mapping_dict1[-1])
                            cycle640_contrast[0].append(selected_samples[0].unsqueeze(0).permute(0, 2, 1)) # (1,T,B)
                            cycle640_contrast[0].append(selected_samples[1].unsqueeze(0).permute(0, 2, 1)) # (1,T,B)
                            cycle640_contrast[0].append(selected_samples[2].unsqueeze(0).permute(0, 2, 1)) # (1,T,B)
                            cycle640_contrast[0].append(selected_samples[3].unsqueeze(0).permute(0, 2, 1)) # (1,T,B)
                            cycle640_contrast[0].append(selected_samples[4].unsqueeze(0).permute(0, 2, 1)) # (1,T,B)
                            cycle640_contrast[0].append(selected_samples[5].unsqueeze(0).permute(0, 2, 1)) # (1,T,B)
                            cycle640_contrast[0].append(selected_samples[6].unsqueeze(0).permute(0, 2, 1)) # (1,T,B)
                            cycle640_contrast[0].append(selected_samples[7].unsqueeze(0).permute(0, 2, 1)) # (1,T,B)
                            cycle640_contrast[0].append(selected_samples[8].unsqueeze(0).permute(0, 2, 1)) # (1,T,B)
                            cycle640_contrast[1].append(selected_samples[10].unsqueeze(0).permute(0, 2, 1)) # (1,T,B)
                            cycle640_contrast[1].append(selected_samples[11].unsqueeze(0).permute(0, 2, 1)) # (1,T,B)
                            cycle640_contrast[1].append(selected_samples[12].unsqueeze(0).permute(0, 2, 1)) # (1,T,B)
                            cycle640_contrast[1].append(selected_samples[13].unsqueeze(0).permute(0, 2, 1)) # (1,T,B)
                            cycle640_contrast[1].append(selected_samples[14].unsqueeze(0).permute(0, 2, 1)) # (1,T,B)
                            cycle640_contrast[1].append(selected_samples[15].unsqueeze(0).permute(0, 2, 1)) # (1,T,B)
                            cycle640_contrast[1].append(selected_samples[16].unsqueeze(0).permute(0, 2, 1)) # (1,T,B)
                            cycle640_contrast[1].append(selected_samples[17].unsqueeze(0).permute(0, 2, 1)) # (1,T,B)
                            cycle640_contrast[1].append(selected_samples[18].unsqueeze(0).permute(0, 2, 1)) # (1,T,B)


                        cycle640_pred = torch.cat(cycle640_cls, dim=0)
                        cycle640_real = torch.cat(cycle640_cls_labels, dim = 0)
                        cycle640_real = cycle640_real.to(self.device)
                        trend_loss += self.criterion(cycle640_pred, cycle640_real.long())

                    

                        cycle640_0 = torch.cat(cycle640_contrast[0], dim=0)
                        cycle640_1 = torch.cat(cycle640_contrast[1], dim=0)
                        cycle_loss += contrastive_loss(
                            cycle640_0,
                            cycle640_1,
                            sample_contrastive_loss,
                            hierarchical=True,
                            factor=1.0,
                        )  

                ct_loss = cycle_loss + trend_loss
                factor_relative = 0.125 # 



                loss = patient_loss + trial_loss + sample_loss + observation_loss + factor_relative*ct_loss

                
                loss.backward()
                optimizer.step()
                self.net.update_parameters(self._net)

                cum_loss += loss.item()
                
                if self.after_iter_callback is not None:
                    self.after_iter_callback(self, loss.item())
                    # print(factor_relative*relative_loss)
                n_epoch_iters += 1
                self.n_iters += 1
            scheduler.step(loss) # 
            if interrupted:
                break
            
            cum_loss /= n_epoch_iters
            epoch_loss_list.append(cum_loss)
            if verbose:
                print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
            
            if self.after_epoch_callback is not None:
                linear_f1 = self.after_epoch_callback(self, cum_loss)
                epoch_f1_list.append(linear_f1)

            self.n_epochs += 1
            
        return epoch_loss_list, epoch_f1_list
    
    def eval_with_pooling(self, x, mask=None):
        """ Pooling the representation.

        """
        out = self.net(x, mask)
        # representation shape: B x O x Co ---> B x 1 x Co ---> B x Co
        out = F.max_pool1d(
            out.transpose(1, 2),
            kernel_size=out.size(1),
        ).transpose(1, 2)
        out = out.squeeze(1)
        return out
    
    def encode(self, X, mask=None, batch_size=None):
        """ Compute representations using the model.
        
        Args:
            X (numpy.ndarray): The input data. This should have a shape of (n_samples, sample_timestamps, features).
            mask (str): The mask used by encoder can be specified with this parameter. Check masking functions in encoder.py.
            batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.
            
        Returns:
            repr: The representations for data.
        """
        assert self.net is not None, 'please train or load a net first'
        assert X.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        # n_samples, ts_l, _ = data.shape

        org_training = self.net.training
        self.net.eval()
        
        dataset = TensorDataset(torch.from_numpy(X).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)
        
        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0].to(self.device)
                # print(next(self.net.parameters()).device)
                # print(x.device)
                out = self.eval_with_pooling(x, mask)
                output.append(out)
                
            output = torch.cat(output, dim=0)
            
        self.net.train(org_training)
        # return output.numpy()
        return output.cpu().numpy()

    def save(self, fn):
        """ Save the model to a file.
        
        Args:
            fn (str): filename.
        """
        torch.save(self.net.state_dict(), fn)
    
    def load(self, fn):
        """ Load the model from a file.
        
        Args:
            fn (str): filename.
        """
        # state_dict = torch.load(fn, map_location=self.device)
        state_dict = torch.load(fn)
        self.net.load_state_dict(state_dict)
    
