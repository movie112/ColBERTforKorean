# colbert/utils/parser.py 
'''
import colbert.utils.distributed as distributed--
from colbert.utils.runs import Run--
from colbert.utils.utils import print_message, timestamp, create_directory--
'''

class Arguments():
    def __init__(self, description):
        self.parser = ArgumentParser(description=description)
        self.checks = []

        self.add_argument('--root', dest='root', default='/root/kolbert/experiments20')
        self.add_argument('--experiment', dest='experiment', default='MSMARCO-psg')
        self.add_argument('--run', dest='run', default=Run.name)

        self.add_argument('--local_rank', dest='rank', default=-1, type=int)

    def add_model_parameters(self):
        # Core Arguments
        self.add_argument('--similarity', dest='similarity', default='cosine', choices=['cosine', 'l2'])
        self.add_argument('--dim', dest='dim', default=128, type=int)
        self.add_argument('--query_maxlen', dest='query_maxlen', default=32, type=int)
        self.add_argument('--doc_maxlen', dest='doc_maxlen', default=180, type=int)

        # Filtering-related Arguments'../movie/data/toptop.tsv'
        self.add_argument('--mask-punctuation', dest='mask_punctuation', default=True, action='store_true')

    def add_model_inference_parameters(self):
        self.add_argument('--checkpoint', dest='checkpoint', default=True, type=str)
        self.add_argument('--bsize', dest='bsize', default=128, type=int)
        self.add_argument('--amp', dest='amp', default=True, action='store_true')
        
    def add_ranking_input(self):
        self.add_argument('--queries', dest='queries', default=True, type=str)
        self.add_argument('--collection', dest='collection', default=True, type=str)
        self.add_argument('--qrels', dest='qrels', default=True, type=str)
# '/home/dilab/movie/data/'
    def add_reranking_input(self):
        self.add_ranking_input()
        self.add_argument('--topk', dest='topK', default=True, type=str)
        self.add_argument('--shortcircuit', dest='shortcircuit', default=False, action='store_true')

    def add_argument(self, *args, **kw_args):
        return self.parser.add_argument(*args, **kw_args)

    def check_arguments(self, args):
        for check in self.checks:
            check(args)

    def parse(self):
        args = self.parser.parse_args()
        self.check_arguments(args)

        args.input_arguments = copy.deepcopy(args)

        args.nranks, args.distributed = distributed_init(args.rank)

        args.nthreads = int(max(os.cpu_count(), faiss.omp_get_max_threads()) * 0.8)
        args.nthreads = max(1, args.nthreads // args.nranks)

        if args.nranks > 1:
            print_message(f"#> Restricting number of threads for FAISS to {args.nthreads} per process",
                          condition=(args.rank == 0))
            faiss.omp_set_num_threads(args.nthreads)

        Run.init(args.rank, args.root, args.experiment, args.run)
        Run._log_args(args)
        Run.info(args.input_arguments.__dict__, '\n')

        return args