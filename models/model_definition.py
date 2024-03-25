
def model_definition(*, args, opt, train_loader, val_loader):
    if args.task_name=="meiyan":
        if args.mode==0:
            from models.meiyan_baseline import meiyan_baseline
            model = meiyan_baseline(args=args,opt=opt,train_loader=train_loader,val_loader=val_loader)
        elif args.mode==1:
            from models.PSNR_calculate import PSNR_calculate
            model = PSNR_calculate(args=args,opt=opt,train_loader=train_loader,val_loader=val_loader)
    elif args.task_name=="forgery":
        from models.forgery_SAM import forgery_SAM
        model = forgery_SAM(args=args, opt=opt, train_loader=train_loader, val_loader=val_loader)
    elif args.task_name=="imuge":
        from models.imuge_retrain.imuge_main import imuge_main
        model = imuge_main(args=args, opt=opt, train_loader=train_loader, val_loader=val_loader)

    else:
        raise NotImplementedError("不支持的mode，请检查！")

    return model