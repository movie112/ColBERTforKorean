def train(args):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if args.distributed:
        torch.cuda.manual_seed_all(SEED)

    if args.distributed:
        assert args.bsize % args.nranks == 0, (args.bsize, args.nranks)
        assert args.accumsteps == 1
        args.bsize = args.bsize // args.nranks

        print("Using args.bsize =", args.bsize, "(per process) and args.accumsteps =", args.accumsteps)

    if args.lazy:
        reader = LazyBatcher(args, (0 if args.rank == -1 else args.rank), args.nranks)
    else:
        reader = EagerBatcher(args, (0 if args.rank == -1 else args.rank), args.nranks)

    if args.rank not in [-1, 0]:
        torch.distributed.barrier()
    gc.collect()
 
    kolbert = KolBERT(
                                      config=MCONFIG,
                                      query_maxlen=args.query_maxlen,
                                      doc_maxlen=args.doc_maxlen,
                                      dim=args.dim,
                                      similarity_metric=args.similarity,
                                      mask_punctuation=args.mask_punctuation
                                      )

    gc.collect()
    if args.checkpoint is not None:
        assert args.resume_optimizer is False, "TODO: This would mean reload optimizer too."
        print_message(f"#> Starting from checkpoint {args.checkpoint} -- but NOT the optimizer!")

        checkpoint = torch.load(args.checkpoint, map_location='cpu')

        try:
            kolbert.load_state_dict(checkpoint['model_state_dict'])
        except:
            print_message("[WARNING] Loading checkpoint with strict=False")
            kolbert.load_state_dict(checkpoint['model_state_dict'], strict=False)

    if args.rank == 0:
        torch.distributed.barrier()
    gc.collect()
    kolbert = kolbert.to(DEVICE)
    kolbert.train()
    gc.collect()
    gc.collect()

    if args.distributed:
        kolbert = torch.nn.parallel.DistributedDataParallel(kolbert, device_ids=[args.rank],
                                                            output_device=args.rank,
                                                            find_unused_parameters=True)

    optimizer = AdamW(filter(lambda p: p.requires_grad, kolbert.parameters()), lr=args.lr, eps=1e-8)
    optimizer.zero_grad()

    amp = MixedPrecisionManager(args.amp)
    criterion = nn.CrossEntropyLoss()
    labels = torch.zeros(args.bsize, dtype=torch.long, device=DEVICE)

    start_time = time.time()
    train_loss = 0.0

    start_batch_idx = 0
    
    gc.collect()
    gc.collect()

    if args.resume:
        assert args.checkpoint is not None
        start_batch_idx = checkpoint['batch']

        reader.skip_to_batch(start_batch_idx, checkpoint['arguments']['bsize'])
        
    pos_lst = []
    neg_lst = []
    loss_lst = []
    for batch_idx, BatchSteps in zip(range(start_batch_idx, args.maxsteps), reader):
        this_batch_loss = 0.0

        for queries, passages in BatchSteps:
            with amp.context():   
                scores = kolbert(queries, passages).view(2, -1).permute(1, 0) ###
                loss = criterion(scores, labels[:scores.size(0)])
                loss = loss / args.accumsteps

            if args.rank < 1:
                # round(scores[:, 0].mean().item(), 2), round(scores[:, 1].mean().item(), 2)
                if batch_idx % 50 == 0:
                    print_progress(scores)
                    positive_avg, negative_avg = round(scores[:, 0].mean().item(), 2), round(scores[:, 1].mean().item(), 2)##
                    pos_lst.append(positive_avg)
                    neg_lst.append(negative_avg)

            amp.backward(loss)

            train_loss += loss.item()
            this_batch_loss += loss.item()

        amp.step(kolbert, optimizer)

        if args.rank < 1:
            avg_loss = train_loss / (batch_idx+1)

            num_examples_seen = (batch_idx - start_batch_idx) * args.bsize * args.nranks
            elapsed = float(time.time() - start_time)

            log_to_mlflow = (batch_idx % 20 == 0)
            Run.log_metric('train/avg_loss', avg_loss, step=batch_idx, log_to_mlflow=log_to_mlflow)
            Run.log_metric('train/batch_loss', this_batch_loss, step=batch_idx, log_to_mlflow=log_to_mlflow)
            Run.log_metric('train/examples', num_examples_seen, step=batch_idx, log_to_mlflow=log_to_mlflow)
            Run.log_metric('train/throughput', num_examples_seen / elapsed, step=batch_idx, log_to_mlflow=log_to_mlflow)
            if batch_idx % 50 == 0:
                print_message(batch_idx, avg_loss)
                loss_lst.append(avg_loss)
            manage_checkpoints(args, kolbert, optimizer, batch_idx+1)
    return pos_lst, neg_lst, loss_lst