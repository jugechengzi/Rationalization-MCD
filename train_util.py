import torch
import torch.nn.functional as F
import torch.nn as nn

from metric import get_sparsity_loss, get_continuity_loss, computer_pre_rec
import numpy as np
import math






def train_sp_norm(model, optimizer, dataset, device, args,writer_epoch,grad,grad_loss):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    train_sp = []
    batch_len=len(dataset)
    for (batch, (inputs, masks, labels)) in enumerate(dataset):

        optimizer.zero_grad()

        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        # rationales, cls_logits
        rationales, logits = model(inputs, masks)

        # computer loss
        cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)

        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales[:, :, 1], masks, args.sparsity_percentage)

        sparsity = (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item()
        train_sp.append(
            (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item())

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales[:, :, 1])

        loss = cls_loss + sparsity_loss + continuity_loss
        # update gradient
        if args.dis_lr==1:
            if sparsity==0:
                lr_lambda=1
            else:
                lr_lambda=sparsity
            if lr_lambda<0.05:
                lr_lambda=0.05
            optimizer.param_groups[1]['lr'] = optimizer.param_groups[0]['lr'] * lr_lambda
        elif args.dis_lr == 0:
            pass
        else:
            optimizer.param_groups[1]['lr'] = optimizer.param_groups[0]['lr'] / args.dis_lr


        loss.backward()


        optimizer.step()

        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()
        cls_l += cls_loss.cpu().item()
        spar_l += sparsity_loss.cpu().item()
        cont_l += continuity_loss.cpu().item()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    writer_epoch[0].add_scalar('cls', cls_l, writer_epoch[1])
    writer_epoch[0].add_scalar('spar_l', spar_l, writer_epoch[1])
    writer_epoch[0].add_scalar('cont_l', cont_l, writer_epoch[1])
    writer_epoch[0].add_scalar('train_sp', np.mean(train_sp), writer_epoch[1])
    return precision, recall, f1_score, accuracy


class JS_DIV(nn.Module):
    def __init__(self):
        super(JS_DIV, self).__init__()
        self.kl_div=nn.KLDivLoss(reduction='batchmean',log_target=True)
    def forward(self,p,q):
        p_s=F.softmax(p,dim=-1)
        q_s=F.softmax(q,dim=-1)
        p_s, q_s = p_s.view(-1, p_s.size(-1)), q_s.view(-1, q_s.size(-1))
        m = (0.5 * (p_s + q_s)).log()
        return 0.5 * (self.kl_div(m, p_s.log()) + self.kl_div(m, q_s.log()))


def train_independent_causal(model, optimizer, dataset, device, args,writer_epoch):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    train_sp = []
    batch_len=len(dataset)
    for (batch, (inputs, masks, labels)) in enumerate(dataset):

        optimizer.zero_grad()

        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        # rationales, cls_logits
        rationales, logits = model(inputs, masks)

        #full_text logits
        full_text_logits=model.train_one_step(inputs, masks)

        # computer loss
        cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)
        full_text_cls_loss=args.cls_lambda * F.cross_entropy(full_text_logits, labels)

        #jsd
        jsd_func=JS_DIV()
        jsd_loss=jsd_func(logits,full_text_logits)

        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales[:, :, 1], masks, args.sparsity_percentage)

        sparsity = (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item()
        train_sp.append(
            (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item())

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales[:, :, 1])

        loss = cls_loss + sparsity_loss + continuity_loss + args.div_lambda*jsd_loss + args.x_lambda*full_text_cls_loss
        # update gradient
        loss.backward()
        optimizer.step()

        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()
        cls_l += cls_loss.cpu().item()
        spar_l += sparsity_loss.cpu().item()
        cont_l += continuity_loss.cpu().item()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    writer_epoch[0].add_scalar('cls', cls_l, writer_epoch[1])
    writer_epoch[0].add_scalar('spar_l', spar_l, writer_epoch[1])
    writer_epoch[0].add_scalar('cont_l', cont_l, writer_epoch[1])
    writer_epoch[0].add_scalar('train_sp', np.mean(train_sp), writer_epoch[1])
    return precision, recall, f1_score, accuracy


def train_decouple_causal(model, opt_gen,opt_pred, dataset, device, args,writer_epoch):

    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    train_sp = []
    batch_len=len(dataset)
    for (batch, (inputs, masks, labels)) in enumerate(dataset):

        opt_gen.zero_grad()
        opt_pred.zero_grad()
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        # rationales, cls_logits
        rationales=model.get_rationale(inputs, masks)

        detach_logits = model.detach_gen_pred(inputs, masks,rationales)
        forward_logit=model.pred_forward_logit(inputs, masks,rationales)



        #full_text logits
        full_text_logits=model.train_one_step(inputs, masks)

        # computer loss
        cls_loss = args.cls_lambda * F.cross_entropy(detach_logits, labels)
        full_text_cls_loss=args.cls_lambda * F.cross_entropy(full_text_logits, labels)

        #jsd
        jsd_func=JS_DIV()
        jsd_loss=jsd_func(forward_logit,full_text_logits)

        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales[:, :, 1], masks, args.sparsity_percentage)

        sparsity = (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item()
        train_sp.append(
            (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item())

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales[:, :, 1])

        # loss = cls_loss + sparsity_loss + continuity_loss + args.div_lambda*jsd_loss + args.x_lambda*full_text_cls_loss
        # # update gradient
        # loss.backward()
        # optimizer.step()
        pred_loss=cls_loss+args.x_lambda*full_text_cls_loss
        gen_loss=sparsity_loss + continuity_loss + args.div_lambda*jsd_loss



        opt_gen.zero_grad()
        gen_loss.backward(retain_graph=True)
        opt_gen.step()

        opt_pred.zero_grad()
        pred_loss.backward()
        opt_pred.step()





        cls_soft_logits = torch.softmax(detach_logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()
        cls_l += cls_loss.cpu().item()
        spar_l += sparsity_loss.cpu().item()
        cont_l += continuity_loss.cpu().item()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    writer_epoch[0].add_scalar('cls', cls_l, writer_epoch[1])
    writer_epoch[0].add_scalar('spar_l', spar_l, writer_epoch[1])
    writer_epoch[0].add_scalar('cont_l', cont_l, writer_epoch[1])
    writer_epoch[0].add_scalar('train_sp', np.mean(train_sp), writer_epoch[1])
    return precision, recall, f1_score, accuracy

def train_decouple_causal2(model, opt_gen,opt_pred, dataset, device, args,writer_epoch):

    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    js=0
    train_sp = []
    batch_len=len(dataset)
    for (batch, (inputs, masks, labels)) in enumerate(dataset):

        opt_gen.zero_grad()
        opt_pred.zero_grad()
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)




        #train classification
        rationales = model.get_rationale(inputs, masks)
        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales[:, :, 1], masks, args.sparsity_percentage)

        sparsity = (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item()
        train_sp.append(
            (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item())

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales[:, :, 1])
        if args.gen_acc == 0:
            forward_logit = model.pred_forward_logit(inputs, masks, torch.detach(rationales))
        elif args.gen_acc==1:
            forward_logit = model.pred_forward_logit(inputs, masks, rationales)
        else:
            print('wrong gen acc')




        full_text_logits = model.train_one_step(inputs, masks)

        cls_loss = args.cls_lambda * F.cross_entropy(forward_logit, labels)
        full_text_cls_loss = args.cls_lambda * F.cross_entropy(full_text_logits, labels)

        if args.gen_sparse==1:
            classification_loss=cls_loss+args.x_lambda*full_text_cls_loss+sparsity_loss + continuity_loss
        elif args.gen_sparse==0:
            classification_loss = cls_loss + args.x_lambda * full_text_cls_loss
        else:
            print('gen sparse wrong')
        classification_loss.backward()

        opt_pred.step()
        opt_pred.zero_grad()
        if args.gen_acc==1:
            opt_gen.step()
            opt_gen.zero_grad()
        elif args.gen_sparse==1:
            opt_gen.step()
            opt_gen.zero_grad()
        else:
            pass


        # train rationale with sparsity, continuity, js-div
        opt_gen.zero_grad()
        name1=[]
        name2=[]
        name3=[]
        for idx,p in model.cls.named_parameters():
            if p.requires_grad==True:
                name1.append(idx)
                p.requires_grad=False
        for idx,p in model.cls_fc.named_parameters():
            if p.requires_grad == True:
                name2.append(idx)
                p.requires_grad = False
        if args.model_type!='sp':
            for idx,p in model.layernorm2.named_parameters():
                if p.requires_grad == True:
                    name3.append(idx)
                    p.requires_grad = False
        # print('name1={},name2={},name3={}'.format(len(name1),len(name2),len(name3)))

        rationales = model.get_rationale(inputs, masks)

        forward_logit = model.pred_forward_logit(inputs, masks, rationales)

        full_text_logits = model.train_one_step(inputs, masks)
        if args.div=='js':
            jsd_func = JS_DIV()
            jsd_loss = jsd_func(forward_logit, full_text_logits)
        elif args.div=='kl':
            jsd_loss=nn.functional.kl_div(F.softmax(forward_logit,dim=-1).log(), F.softmax(full_text_logits,dim=-1), reduction='batchmean')

        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales[:, :, 1], masks, args.sparsity_percentage)

        sparsity = (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item()
        train_sp.append(
            (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item())

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales[:, :, 1])

        gen_loss = sparsity_loss + continuity_loss + args.div_lambda * jsd_loss

        gen_loss.backward()
        opt_gen.step()
        opt_gen.zero_grad()
        n1=0
        n2=0
        n3=0
        for idx,p in model.cls.named_parameters():
            if idx in name1:
                p.requires_grad=True
                n1+=1
        for idx,p in model.cls_fc.named_parameters():
            if idx in name2:
                p.requires_grad = True
                n2 += 1
        if args.model_type != 'sp':
            for idx,p in model.layernorm2.named_parameters():
                if idx in name3:
                    p.requires_grad = True
                    n3 += 1
        # print('recover name1={},name2={},name3={}'.format(n1, n2, n3))







        cls_soft_logits = torch.softmax(forward_logit, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()
        cls_l += cls_loss.cpu().item()
        spar_l += sparsity_loss.cpu().item()
        cont_l += continuity_loss.cpu().item()
        js+=jsd_loss.cpu().item()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    writer_epoch[0].add_scalar('cls', cls_l, writer_epoch[1])
    writer_epoch[0].add_scalar('spar_l', spar_l, writer_epoch[1])
    writer_epoch[0].add_scalar('cont_l', cont_l, writer_epoch[1])
    writer_epoch[0].add_scalar('js', js, writer_epoch[1])
    writer_epoch[0].add_scalar('train_sp', np.mean(train_sp), writer_epoch[1])
    return precision, recall, f1_score, accuracy

def train_decouple_causal_samerattionale(model, opt_gen,opt_pred, dataset, device, args,writer_epoch):

    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    js=0
    train_sp = []
    batch_len=len(dataset)
    for (batch, (inputs, masks, labels)) in enumerate(dataset):

        opt_gen.zero_grad()
        opt_pred.zero_grad()
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)




        #train classification
        rationales = model.get_rationale(inputs, masks)
        forward_logit = model.pred_forward_logit(inputs, masks, torch.detach(rationales))

        # sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
        #     rationales[:, :, 1], masks, args.sparsity_percentage)
        #
        # sparsity = (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item()
        # train_sp.append(
        #     (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item())
        #
        # continuity_loss = args.continuity_lambda * get_continuity_loss(
        #     rationales[:, :, 1])





        full_text_logits = model.train_one_step(inputs, masks)

        cls_loss = args.cls_lambda * F.cross_entropy(forward_logit, labels)
        full_text_cls_loss = args.cls_lambda * F.cross_entropy(full_text_logits, labels)


        classification_loss = cls_loss + args.x_lambda * full_text_cls_loss

        classification_loss.backward()

        opt_pred.step()
        opt_pred.zero_grad()




        # train rationale with sparsity, continuity, js-div
        opt_gen.zero_grad()
        name1=[]
        name2=[]
        name3=[]
        for idx,p in model.cls.named_parameters():
            if p.requires_grad==True:
                name1.append(idx)
                p.requires_grad=False
        for idx,p in model.cls_fc.named_parameters():
            if p.requires_grad == True:
                name2.append(idx)
                p.requires_grad = False
        for idx,p in model.layernorm2.named_parameters():
            if p.requires_grad == True:
                name3.append(idx)
                p.requires_grad = False


        # rationales = model.get_rationale(inputs, masks)

        forward_logit = model.pred_forward_logit(inputs, masks, rationales)

        full_text_logits = model.train_one_step(inputs, masks)
        if args.div=='js':
            jsd_func = JS_DIV()
            jsd_loss = jsd_func(forward_logit, full_text_logits)
        elif args.div=='kl':
            jsd_loss=nn.functional.kl_div(F.softmax(forward_logit,dim=-1).log(), F.softmax(full_text_logits,dim=-1), reduction='batchmean')

        # sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
        #     rationales[:, :, 1], masks, args.sparsity_percentage)
        #
        # sparsity = (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item()
        # train_sp.append(
        #     (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item())
        #
        # continuity_loss = args.continuity_lambda * get_continuity_loss(
        #     rationales[:, :, 1])
        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales[:, :, 1], masks, args.sparsity_percentage)

        sparsity = (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item()
        train_sp.append(
            (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item())

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales[:, :, 1])

        gen_loss = sparsity_loss + continuity_loss + args.div_lambda * jsd_loss

        gen_loss.backward()
        opt_gen.step()
        opt_gen.zero_grad()

        n1=0
        n2=0
        n3=0
        name1_re=[]
        name2_re = []
        name3_re = []
        for idx,p in model.cls.named_parameters():
            if idx in name1:
                p.requires_grad=True
                n1+=1
                name1_re.append(idx)
        for idx,p in model.cls_fc.named_parameters():
            if idx in name2:
                p.requires_grad = True
                n2 += 1
                name2_re.append(idx)
        for idx,p in model.layernorm2.named_parameters():
            if idx in name3:
                p.requires_grad = True
                n3 += 1
                name3_re.append(idx)








        cls_soft_logits = torch.softmax(forward_logit, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()
        cls_l += cls_loss.cpu().item()
        spar_l += sparsity_loss.cpu().item()
        cont_l += continuity_loss.cpu().item()
        js+=jsd_loss.cpu().item()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    writer_epoch[0].add_scalar('cls', cls_l, writer_epoch[1])
    writer_epoch[0].add_scalar('spar_l', spar_l, writer_epoch[1])
    writer_epoch[0].add_scalar('cont_l', cont_l, writer_epoch[1])
    writer_epoch[0].add_scalar('js', js, writer_epoch[1])
    writer_epoch[0].add_scalar('train_sp', np.mean(train_sp), writer_epoch[1])
    return precision, recall, f1_score, accuracy

def classfy(model, optimizer, dataset, device, args):
    TP = 0
    TN = 0
    FN = 0
    FP = 0

    for (batch, (inputs, masks, labels)) in enumerate(dataset):
        optimizer.zero_grad()

        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        # rationales, cls_logits
        logits = model(inputs, masks)

        # computer loss
        cls_loss =F.cross_entropy(logits, labels)


        loss = cls_loss

        # update gradient
        loss.backward()
        print('yes')
        optimizer.step()
        print('yes2')

        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return precision, recall, f1_score, accuracy



def train_g_skew(model, optimizer, dataset, device, args):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    for (batch, (inputs, masks, labels)) in enumerate(dataset):
        optimizer.zero_grad()
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
        logits=model.g_skew(inputs,masks)[:,0,:]
        cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)
        cls_loss.backward()
        optimizer.step()
        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return precision, recall, f1_score, accuracy




