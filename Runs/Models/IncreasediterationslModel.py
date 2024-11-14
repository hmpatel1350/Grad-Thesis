import time

import torch

from Models.DynamicalSystemModel import DynamicalSystemModel

class IncreasedIterationsModel(DynamicalSystemModel):
    def train(self):
        only_gpu = 0.0
        start_time = time.time()
        change = self.config['epochs'] / self.iterations
        H_cache = []

        # We pad H with zeros when i or j are out of bounds
        def getH(batch_idx):
            return H_cache[batch_idx]

        for epoch in range(self.config['epochs']):
            iterations_now = 1 + (epoch * self.iterations) // (self.config['epochs'])

            # Precompute for the current epoch
            for batch_idx, (P0, P1, attention_mask) in enumerate(self.dataloader):
                P0_repeated = P0.unsqueeze(1).repeat(1, iterations_now, 1, 1)
                P1_repeated = P1.unsqueeze(1).repeat(1, iterations_now, 1, 1)

                start_gpu = time.time()
                self.optimizer.zero_grad()
                end_gpu = time.time()
                only_gpu += (end_gpu - start_gpu)

                loss = 0.0
                # a 2 dimension list of m by n of None values
                # this will be used to store the hidden states
                if epoch == 0:
                    H_initial = torch.zeros(P0.shape[0], iterations_now, self.path_length, self.h_size, device=self.accelerator.device)
                    H_cache.append(H_initial)

                H_old = getH(batch_idx)

                zeros_i = torch.zeros(H_old.shape[0], 1, H_old.shape[2], H_old.shape[3], device=self.accelerator.device)
                H_i = torch.cat((zeros_i, H_old), dim=1)
                if (epoch % change != 0 or epoch == 0):
                    H_i = H_i[:, :-1, :, :]

                zeros_j = torch.zeros(H_old.shape[0], H_old.shape[1], 1, H_old.shape[3], device=self.accelerator.device)
                H_j = torch.cat((zeros_j, H_old), dim=2)
                H_j = H_j[:, :, :-1, :]
                if (epoch % change == 0 and epoch != 0):
                    H_j = torch.cat((H_j, zeros_i), dim=1)

                P0_reshaped = P0_repeated.reshape(-1, self.data_dimension)
                H_i_reshaped = H_i.reshape(-1, self.h_size)
                H_j_reshaped = H_j.reshape(-1, self.h_size)

                start_gpu = time.time()
                out, h = self.model(P0_reshaped, H_i_reshaped, H_j_reshaped)
                end_gpu = time.time()
                only_gpu += (end_gpu - start_gpu)

                out_reshaped = out.reshape(P0_repeated.shape[0], P0_repeated.shape[1], P0_repeated.shape[2],
                                           P0_repeated.shape[3])
                loss = 0.0

                for i in range(iterations_now):
                    loss += (i + 1) * self.loss_fn(out_reshaped[:, i, :, :], P1_repeated[:, i, :, :])

                self.wandb_run.log({f'{self.name}_loss': loss})

                start_gpu = time.time()
                loss.backward()
                self.optimizer.step()
                end_gpu = time.time()
                only_gpu += (end_gpu - start_gpu)

                H_cache[batch_idx] = h.reshape(P0.shape[0], iterations_now, self.path_length, self.h_size).detach()

        end_time = time.time()
        elapsed_time = end_time - start_time
        self.elapsed_time = elapsed_time
        self.only_gpu = only_gpu
        return elapsed_time, only_gpu
