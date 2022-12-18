"""
Evaluation process of real datasets

"""

import os
import pickle

from metrics import Metrics
from utils import check_folder


class Evaluation(object):
    def __init__(self, args):
        # set parameters
        self.domain = args.domain
        self.dataroot = args.dataroot
        self.dataset = args.dataset
        self.result_dir = args.result_dir
        self.num_workers = args.num_workers
        self.image_size = args.image_size
        self.eval = {}

        # config directories
        self.evaluation_dir = os.path.join(self.result_dir, self.dataset)
        self.evaluation_dir = os.path.join(self.evaluation_dir, "REAL_evaluation")
        check_folder(self.evaluation_dir)

        # set evaluation parameters
        self.eval['is_score'] = []
        self.eval['fid_score'] = []

    # evaluate real data
    def evaluate_reals(self):
        """
        Real image datasets evaluation process implementation

        """

        print("Evaluating real data...")
        if self.domain == "image":
            argsM = {'dataroot': self.dataroot, 'dataset': self.dataset, 'image_size': self.image_size,
                     'num_workers': self.num_workers}
            metrics = Metrics(argsM=argsM, domain=self.domain, batch_size=50, sample_size=50000, REALS=True)
        else:
            argsM = {'dataroot': self.dataroot, 'dataset': self.dataset}
            metrics = Metrics(argsM=argsM, domain=self.domain, batch_size=10, sample_size=1000, REALS=True)
        is_mean, is_std, fid_score = metrics.calculate_scores()
        is_score = {"Real Data": [is_mean, is_std]}
        fid = {"Real Data": fid_score}
        self.eval['is_score'].append(is_score)
        self.eval['fid_score'].append(fid)
        print("Real data " + self.dataset + " is_score: %.4f \u00B1" % is_mean, is_std)
        print("Real data " + self.dataset + " fid_score: %.4f" % fid_score)
        with open(self.evaluation_dir + "/evaluation.pkl", 'wb') as f:
            pickle.dump(self.eval, f)
