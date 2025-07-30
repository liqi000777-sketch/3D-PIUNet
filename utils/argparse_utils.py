def add_lightning_args(parser):
    parser.add_argument("--max_epochs", default=300, type=int)
    parser.add_argument("--auto_scale_batch_size", default="", type=str)
    parser.add_argument("--resume_from_checkpoint", default=None, type=str)
    parser.add_argument("--devices", default=1, type=int)
    parser.add_argument("--accelerator", default="cpu", type=str)


    return parser