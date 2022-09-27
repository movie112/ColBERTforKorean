def main():
    parser = Arguments(description='Training ColBERT with <query, positive passage, negative passage> triples.')

    parser.add_model_parameters()
    parser.add_model_training_parameters()
    parser.add_training_input()
    parser.add_argument('-f')
    
    args = parser.parse()
    
    labels = torch.zeros(args.bsize, dtype=torch.long, device=DEVICE) #32, torch.int64

    assert args.bsize % args.accumsteps == 0, ((args.bsize, args.accumsteps),
                                               "The batch size must be divisible by the number of gradient accumulation steps.")
    assert args.query_maxlen <= 512
    assert args.doc_maxlen <= 512
    
    args.lazy = args.collection is not None

    with Run.context(consider_failed_if_interrupted=False):
        return train(args)


if __name__ == "__main__":
    pos_lst, neg_lst, loss_lst = main()
    
    print('positive_score_avg :', sum(pos_lst)/len(pos_lst))
    print('negative_score_avg :', sum(neg_lst)/len(neg_lst))