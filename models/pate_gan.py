# Copyright 2019 RBC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# pate_gan.py implements the PATE_GAN generative model to generate private synthetic data
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import numpy as np
import pandas as pd
import math
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score


from utils.architectures import Generator, Discriminator
from utils.helper import weights_init, pate, moments_acc
import csv



class PATE_GAN:
    def __init__(self, leaky, logfile, input_dim, z_dim, num_teachers, target_epsilon, target_delta, conditional=True):
        self.generator = Generator(z_dim, input_dim, conditional, leaky).cuda().double()
        self.student_disc = Discriminator(input_dim, leaky, wasserstein=False).cuda().double()
        self.teacher_disc = [Discriminator(input_dim, leaky, wasserstein=False).cuda().double()
                             for _ in range(num_teachers)]
        self.generator.apply(weights_init)
        self.student_disc.apply(weights_init)
        self.z_dim = z_dim
        self.num_teachers = num_teachers
        for i in range(num_teachers):
            self.teacher_disc[i].apply(weights_init)

        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.conditional = conditional
        self.logfile = logfile # Should be a path to a csv file!

    def train(self, x_train, y_train, x_test, y_test, colnames, scaler, hyperparams):
        csvfile = open(self.logfile, 'w')
        csvwriter = csv.writer(csvfile, delimiter=',')

        best_roc_score = 0
        batch_size = hyperparams.batch_size
        num_teacher_iters = hyperparams.num_teacher_iters
        num_student_iters = hyperparams.num_student_iters
        num_moments = hyperparams.num_moments
        lap_scale = hyperparams.lap_scale
        class_ratios = None
        if self.conditional:
            class_ratios = torch.from_numpy(hyperparams.class_ratios)

        real_label = 1
        fake_label = 0

        alpha = torch.cuda.DoubleTensor([0.0 for _ in range(num_moments)])
        l_list = 1 + torch.cuda.DoubleTensor(range(num_moments))

        criterion = nn.BCELoss()
        optimizer_g = optim.Adam(self.generator.parameters(), lr=hyperparams.lr)
        optimizer_sd = optim.Adam(self.student_disc.parameters(), lr=hyperparams.lr)
        optimizer_td = [optim.Adam(self.teacher_disc[i].parameters(), lr=hyperparams.lr
                                   ) for i in range(self.num_teachers)]

        tensor_data = data_utils.TensorDataset(torch.cuda.DoubleTensor(x_train), torch.cuda.DoubleTensor(y_train))

        train_loader = []
        for teacher_id in range(self.num_teachers):
            start_id = teacher_id * len(tensor_data) / self.num_teachers
            end_id = (teacher_id + 1) * len(tensor_data) / self.num_teachers if teacher_id != (
                    self.num_teachers - 1) else len(tensor_data)

            train_loader.append(data_utils.DataLoader(torch.utils.data.Subset( \
                tensor_data, range(int(start_id), int(end_id))), batch_size=batch_size, shuffle=True))

        steps = 0
        epsilon = 0

        while epsilon < self.target_epsilon:

            # train the teacher discriminators
            for t_2 in range(num_teacher_iters):
                for i in range(self.num_teachers):
                    inputs, categories = None, None
                    for b, data in enumerate(train_loader[i], 0):
                        inputs, categories = data
                        break

                    # train with real
                    optimizer_td[i].zero_grad()
                    label = torch.full((inputs.size()[0],), real_label).cuda()
                    output = self.teacher_disc[i].forward(torch.cat([inputs, categories.unsqueeze(1).double()], dim=1))
                    label = label.unsqueeze(1)
                    label = label.double()
                    err_d_real = criterion(output, label)
                    err_d_real.backward()

                    # train with fake
                    z = torch.Tensor(batch_size, self.z_dim).uniform_(0, 1).cuda()
                    label.fill_(fake_label)

                    if self.conditional:
                        category = torch.multinomial(class_ratios,  inputs.size()[0], replacement=True).unsqueeze(1).cuda().double()
                        fake = self.generator(torch.cat([z.double(), category], dim=1))
                        output = self.teacher_disc[i].forward(torch.cat([fake.detach(), category], dim=1))
                    else:
                        fake = self.generator(z.double())
                        output = self.teacher_disc[i].forward(fake)

                    err_d_fake = criterion(output, label.double())
                    err_d_fake.backward()
                    optimizer_td[i].step()

            # train the student discriminator
            for t_3 in range(num_student_iters):
                z = torch.Tensor(batch_size, self.z_dim).uniform_(0, 1).cuda()

                if self.conditional:
                    category = torch.multinomial(class_ratios,  inputs.size()[0], replacement=True).unsqueeze(1).cuda().double()
                    fake = self.generator(torch.cat([z.double(), category], dim=1))
                    predictions, clean_votes = pate(torch.cat(
                        [fake.detach(), category], dim=1), self.teacher_disc, lap_scale)
                    outputs = self.student_disc.forward(torch.cat([fake.detach(), category], dim=1))
                else:
                    fake = self.generator(z.double())
                    predictions, clean_votes = pate(fake.detach(), self.teacher_disc, lap_scale)
                    outputs = self.student_disc.forward(fake.detach())

                # update the moments
                alpha = alpha + moments_acc(self.num_teachers, clean_votes, lap_scale, l_list)

                # update student

                err_sd = criterion(outputs, predictions)

                optimizer_sd.zero_grad()
                err_sd.backward()
                optimizer_sd.step()

            # train the generator
            optimizer_g.zero_grad()
            z = torch.Tensor(batch_size, self.z_dim).uniform_(0, 1).cuda()
            label = torch.full((inputs.size()[0],), real_label).cuda()

            if self.conditional:
                category = torch.multinomial(class_ratios,  inputs.size()[0], replacement=True).unsqueeze(1).cuda().double()
                fake = self.generator(torch.cat([z.double(), category], dim=1))
                output = self.student_disc(torch.cat([fake, category.double()], dim=1))
            else:
                fake = self.generator(z.double())
                output = self.student_disc.forward(fake)
            label = label.unsqueeze(1)
            label = label.double()
            err_g = criterion(output, label)
            err_g.backward()
            optimizer_g.step()

            # Calculate the current privacy cost
            epsilon = min((alpha - math.log(self.target_delta)) / l_list)
            
            # This chunk to generate, save synthetic, compute ROC.
            if steps % 10 == 0:
                syn_data = self.generate(x_train.shape[0], hyperparams.class_ratios)
                syn_x, syn_y = syn_data[:, :-1], syn_data[:, -1]
                syn_save = scaler.inverse_transform(syn_x)

                mlp = MLPClassifier((32,8), max_iter=1000, random_state=42)
                mlp.fit(syn_x, syn_y)
                pred_y = mlp.predict(x_test)

                roc_score =  roc_auc_score(y_test, pred_y)
                if roc_score > best_roc_score:
                    print(f'Best Roc of {best_roc_score} found, saving....')
                    best_roc_score = roc_score
                    self.save_churn(syn_save, syn_y, colnames, steps, roc_score)



            # Do logging to csvfile
            if steps % 10 == 0:
                csvwriter.writerow([steps, err_sd.item(),err_g.item(), epsilon.item(),roc_score])
            # Log to console
            if steps % 10 == 0:
                print("Step : ", steps, "Loss SD : ", err_sd.item(), "Loss G : ", err_g.item(), "Epsilon : ",
                      epsilon.item(), "ROC_SCORE: ", roc_score)

            steps += 1
        
        # End of training
        print(f'Done training after {steps} Steps and Final epsilon of {epsilon.item()}, achieved Top ROC score of {best_roc_score}!')
        csvfile.close()

    def generate(self, num_rows, class_ratios, batch_size=1000):
        steps = num_rows // batch_size
        synthetic_data = []
        if self.conditional:
            class_ratios = torch.from_numpy(class_ratios)
        for step in range(steps):
            noise = torch.randn(batch_size, self.z_dim).cuda()
            if self.conditional:
                cat = torch.multinomial(class_ratios, batch_size, replacement=True).unsqueeze(1).cuda().double()
                synthetic = self.generator(torch.cat([noise.double(), cat], dim=1))
                synthetic = torch.cat([synthetic, cat], dim=1)

            else:
                synthetic = self.generator(noise.double())

            synthetic_data.append(synthetic.cpu().data.numpy())

        if steps * batch_size < num_rows:
            noise = torch.randn(num_rows - steps * batch_size, self.z_dim).cuda()

            if self.conditional:
                cat = torch.multinomial(class_ratios, num_rows - steps * batch_size, replacement=True).unsqueeze(
                    1).cuda().double()
                synthetic = self.generator(torch.cat([noise.double(), cat], dim=1))
                synthetic = torch.cat([synthetic, cat], dim=1)
            else:
                synthetic = self.generator(noise.double())
            synthetic_data.append(synthetic.cpu().data.numpy())

        return np.concatenate(synthetic_data)


    def update_array(self, indexes, cols = None):
        if cols: colsize = cols
        else: colsize = indexes.max() +1
        b = np.zeros((indexes.size, colsize))
        b[np.arange(indexes.size), indexes] = 1
        return b
        
    def save_marketing(self, syn_save):
    # Some fancy indexing to get the actual synthetic data..
        accepted = np.argmax(syn_save[:,16:21], axis=1)
        education = np.argmax(syn_save[:, 22:27], axis=1)
        marital = np.argmax(syn_save[:, 27:34], axis=1)
        country = np.argmax(syn_save[:, 34:], axis=1)

        syn_save[:,16:21] = self.update_array(accepted, cols=5)
        syn_save[:, 22:27] = self.update_array(education, cols=5)
        syn_save[:, 27:34] = self.update_array(marital, cols=7)
        syn_save[:, 34:] = self.update_array(country, cols=8)

        df1 = pd.DataFrame(syn_save, columns = df.columns.drop(TARGET_VARIABLE))
        df2 = pd.DataFrame(syn_y, columns = [TARGET_VARIABLE])
        df_save = pd.concat([df1,df2], axis =1)
        df_save.to_csv(f'synthetic_{MODEL_NAME}_{DATASET_NAME}_{TARGET_EPSILON}.csv')

    def save_churn(self, syn_save, syn_y, colnames, step, roc_score):
        geography = np.argmax(syn_save[:,8:11], axis=1)
        gender = np.argmax(syn_save[:,11:], axis=1)
        
        syn_save[:,8:11] = self.update_array(geography, cols=3)
        syn_save[:, 11:] = self.update_array(gender, cols=2)
        syn_save[:,4] = np.round(syn_save[:,4]) # num products
        syn_save[:,5] = np.round(np.clip(syn_save[:,5],0,1)) # Has card
        syn_save[:,6] = np.round(np.clip(syn_save[:,6],0,1)) # Is active


        df1 = pd.DataFrame(syn_save, columns = colnames)
        df2 = pd.DataFrame(syn_y, columns = ['Exited'])
        df_save = pd.concat([df1,df2], axis =1)
        df_save.to_csv(f'syn/synthetic_pategan_churn_{step:04}_{roc_score:.3f}.csv')