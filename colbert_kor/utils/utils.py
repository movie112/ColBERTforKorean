def print_message(*s, condition=True):
    s = ' '.join([str(x) for x in s])
    msg = "[{}] {}".format(datetime.datetime.now().strftime("%b %d, %H:%M:%S"), s)

    if condition:
        print(msg, flush=True)

    return msg
def timestamp():
    format_str = "%Y-%m-%d_%H.%M.%S"
    result = datetime.datetime.now().strftime(format_str)
    return result

def save_checkpoint(path, epoch_idx, mb_idx, model, optimizer, arguments=None):
    print(f"#> Saving a checkpoint to {path} ..")

    if hasattr(model, 'module'):
        model = model.module  # extract model from a distributed/data-parallel wrapper

    checkpoint = {}
    checkpoint['epoch'] = epoch_idx
    checkpoint['batch'] = mb_idx
    checkpoint['model_state_dict'] = model.state_dict()
    checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    checkpoint['arguments'] = arguments

    torch.save(checkpoint, path)

def create_directory(path):
    if os.path.exists(path):
        print('\n')
        print_message("#> Note: Output directory", path, 'already exists\n\n')
    else:
        print('\n')
        print_message("#> Creating directory", path, '\n\n')
        os.makedirs(path)

def distributed_init(rank):
    nranks = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE'])
    nranks = max(1, nranks)
    is_distributed = nranks > 1

    if rank == 0:
        print('nranks =', nranks, '\t num_gpus =', torch.cuda.device_count())

    if is_distributed:
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % num_gpus)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    return nranks, is_distributed
def distributed_barrier(rank):
    if rank >= 0:
        torch.distributed_barrier()
# see https://stackoverflow.com/a/45187287
class NullContextManager(object):
    def __init__(self, dummy_resource=None):
        self.dummy_resource = dummy_resource
    def __enter__(self):
        return self.dummy_resource
    def __exit__(self, *args):
        pass
    
def load_checkpoint(path, model, optimizer=None, do_print=True):
    if do_print:
        print_message("#> Loading checkpoint", path, "..")

    if path.startswith("http:") or path.startswith("https:"):
        checkpoint = torch.hub.load_state_dict_from_url(path, map_location='cpu')
    else:
        checkpoint = torch.load(path, map_location='cpu')

    state_dict = checkpoint['model_state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if k[:7] == 'module.':
            name = k[7:]
        new_state_dict[name] = v

    checkpoint['model_state_dict'] = new_state_dict

    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        print_message("[WARNING] Loading checkpoint with strict=False")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if do_print:
        print_message("#> checkpoint['epoch'] =", checkpoint['epoch'])
        print_message("#> checkpoint['batch'] =", checkpoint['batch'])