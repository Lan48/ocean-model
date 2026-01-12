from .base import *
from itertools import accumulate
from bisect import bisect_right


class ReanalyDataset(BaseDataset):
    def __init__(self, args, root=None, split='train') -> None:
        super().__init__(args)

        if root is not None:
            self.root = root
        else:
            self.root = args.data_dir

        self.input_var_list = None
        self.times = None

        self.input_var_list = self.list_dir_fn(self.root)

        if len(args.input_var_list) != 0:
            for v in args.input_var_list:
                if v not in CMIP62GODAS:
                    raise ValueError(f'variable {v} is not contained in CMIP6')
            self.input_var_list = args.input_var_list

        if len(args.var_list_not_used) != 0:
            for v in args.var_list_not_used:
                if v in self.input_var_list:
                    self.input_var_list.remove(v)

        self.input_var_list = sorted(self.input_var_list)

        if 'tauu' in self.input_var_list:
            self.input_var_list.remove('tauu')
        if 'tauv' in self.input_var_list:
            self.input_var_list.remove('tauv')

        self.single_lev_vars = [v for v in self.input_var_list if v in SINGLE_LEVEL_VARS]
        self.multi_lev_vars = [v for v in self.input_var_list if v in MULTY_LEVEL_VARS]
        self.atmo_var_list = sorted(args.atmo_var_list)

        self.mix = len(self.single_lev_vars) != 0 and len(self.multi_lev_vars) != 0

        self.split = split

        self.load_times()

        self.num_times = len(self.times)
        self.num_var = len(self.input_var_list)

        self.input_steps = args.input_steps
        self.predict_steps = args.predict_steps

        self.max_t = args.max_t
        self.valid_lead_t = args.valid_lead_t

        self.num_samples = self.num_times - self.input_steps - self.predict_steps - (self.max_t-1) + 1

        if args.lead_t_weight:
            t_weight = np.linspace(args.max_t_weight, args.min_t_weight, self.max_t)
            self.t_weight = t_weight / t_weight.sum()
        else:
            self.t_weight = np.ones(self.max_t) / self.max_t

        self.split_by = args.train_val_test_split_by

        if args.do_cache:
            self.cache_samples()

    def load_times(self, all_time=False, check=False):
        self.times = sorted([t.split('.')[0] for t in
                             self.list_dir_fn(os.path.join(self.root, self.input_var_list[0]))],
                            key=lambda x: int(x.split('_')[0]) * 12 + int(x.split('_')[1]))
        start_year = getattr(self.args, 'start_year', None)
        end_year = getattr(self.args, 'end_year', None)
        st_idx = 0
        end_idx = len(self.times)

        if self.split == 'train' and self.args.train_val_split_year is not None:
            end_year = self.args.train_val_split_year
        elif self.split == 'valid' and self.args.train_val_split_year is not None:
            start_year = self.args.train_val_split_year

        if not all_time:
            if start_year is not None:
                for i in range(len(self.times)):
                    if int(self.times[i].split('_')[0]) == start_year:
                        break
                st_idx = i
            if end_year is not None:
                for i in range(len(self.times)):
                    if int(self.times[i].split('_')[0]) == end_year:
                        break
                end_idx = i
        
        self.times = self.times[st_idx:end_idx]

        if check:
            self.check_times()

    def random_lead_t(self):
        # print(self.max_t, self.t_weight.shape)
        return int(np.random.choice(np.arange(self.max_t), size=1, p=self.t_weight))

    def get_input_var_list_cmip6(self):
        return self.input_var_list

    def get_input_var_list_godas(self):
        return [CMIP62GODAS[v] for v in self.input_var_list]

    def get_var_index(self, v_name):
        if self.mix:
            if v_name in self.single_lev_vars:
                return self.single_lev_vars.index(v_name)
            else:
                return self.multi_lev_vars.index(v_name)

        return self.input_var_list.index(v_name)

    def get_ocean_vars(self, base_path, time_range):
        data = []
        for v in self.input_var_list:
            if v in self.single_lev_vars:
                combine_fn = torch.stack
            else:
                combine_fn = torch.cat
            data.append(combine_fn([
                self.get_data(os.path.join(base_path, v, f'{self.times[i]}.npy'))
                for i in time_range
            ]))
        return torch.cat(data).float()
    
    def get_label_values(self, base_path, time_range):
        data = [torch.cat([
            self.get_data(os.path.join(base_path, v, f'{self.times[i]}.npy')).unsqueeze(0)
            if v in self.single_lev_vars else self.get_data(os.path.join(base_path, v, f'{self.times[i]}.npy'))
            for v in self.input_var_list
        ]) for i in time_range]
        if len(data) == 1:
            return data[0]
        else:
            return torch.stack(data).float()

    def get_data_atmo(self, path):
        data = np.load(path)
        return torch.from_numpy(data)
    
    def get_atmo_vars(self, base_path, time_range):
        data = [torch.cat([
            self.get_data_atmo(os.path.join(base_path, v, f'{self.times[i]}.npy')).unsqueeze(0)
            for i in time_range
        ]) for v in self.atmo_var_list]
        return torch.cat(data).float()
        
    def __len__(self):
        if self.split == 'train':
            return self.max_t * self.num_samples
        else:
            return self.num_samples

    def __getitem__(self, index):

        if self.split == 'train':
            lead_t = index % self.max_t
            index = index // self.max_t
        else:
            lead_t = self.valid_lead_t - 1

        start_month = int(self.times[index].split('_')[-1])

        # lead_t = self.random_lead_t()
        inputs_range = range(index, index+self.input_steps)
        labels_range = range(index+self.input_steps+lead_t, index+self.input_steps+lead_t+self.predict_steps)

        ocean_vars = self.get_ocean_vars(self.root, inputs_range)
        atmo_vars = self.get_atmo_vars(self.root, inputs_range)
        labels = self.get_label_values(self.root, labels_range)

        ret = {
            'ocean_vars': ocean_vars.float(),
            'atmo_vars': atmo_vars.float(),
            'labels': labels.float(),
            'lead_time': torch.tensor(lead_t),
            'start_month': torch.tensor(start_month)
        }

        mask_generator = getattr(self, 'mask_generator', None)
        if mask_generator is not None:
            ret['mask'] = mask_generator()

        return ret


class ReanalyCombinedDataset(BaseDataset):
    def __init__(self, args, root=None, split='train') -> None:
        super().__init__(args)

        if root is not None:
            self.root = root
        else:
            self.root = args.data_dir
            
        if isinstance(self.root, str):
            self.root = self.root.split(',')

        self.split = split

        self.datasets = []
        self.lengths = []
        for p in self.root:
            ds = ReanalyDataset(args, p, split)
            self.datasets.append(ds)
            self.lengths.append(len(ds))
        
        self.lengths_cumsum = list(accumulate(self.lengths))


    def get_input_var_list_cmip6(self):
        return self.datasets[0].get_input_var_list_cmip6()

    def get_input_var_list_godas(self):
        return self.datasets[0].get_input_var_list_godas()

    def get_var_index(self, v_name):
        return self.datasets[0].get_var_index(v_name)

    def __len__(self):
        return self.lengths_cumsum[-1]

    def __getitem__(self, index):

        ds_idx = bisect_right(self.lengths_cumsum, index)
        if ds_idx == 0:
            sample_idx = index
        else:
            sample_idx = index - self.lengths_cumsum[ds_idx-1]
        
        return self.datasets[ds_idx][sample_idx]