# -Multi_GPU_sample_code_with_Apex


This is a fast document for modifying code into Apex and multiple GPUs for training.

### 0. The command for python training
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py


### 1. Add one more argument for GPU assignment
parser.add_argument('--local_rank', type=int, default=0，help='node rank for distributed training')
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group('nccl',init_method='env://')
device = torch.device(f'cuda:{args.local_rank}')

### 2. Assiging the dataset
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=..., sampler=train_sampler)

### 3. changing model into parallel one
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])


### 4. After installing Apex (see their Apex github site, very easy to install ), add these lines in your code

from apex import amp

model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
model = DistributedDataParallel(model, delay_allreduce=True)

with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()

### 5. save the model, only assigning one GPU, otherwise it will clash.
if arg.locak_rank == 0：
    torch.save(xxxx)


## demo code

### main.py
import torch
import argparse
import torch.distributed as dist

from apex.parallel import convert_syncbn_model
from apex.parallel import DistributedDataParallel

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
args = parser.parse_args()

dist.init_process_group(backend='nccl')
torch.cuda.set_device(args.local_rank)

train_dataset = ...
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=..., sampler=train_sampler)

model = ...
#
model = convert_syncbn_model(model)
#
model, optimizer = amp.initialize(model, optimizer)
#
model = DistributedDataParallel(model, device_ids=[args.local_rank])

optimizer = optim.SGD(model.parameters())

for epoch in range(100):
   for batch_idx, (data, target) in enumerate(train_loader):
      images = images.cuda(non_blocking=True)
      target = target.cuda(non_blocking=True)
      ...
      output = model(images)
      loss = criterion(output, target)
      optimizer.zero_grad()
      with amp.scale_loss(loss, optimizer) as scaled_loss:
         scaled_loss.backward()
      optimizer.step()
