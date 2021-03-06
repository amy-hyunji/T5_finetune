import numpy as np

from torch.utils.data import Dataset, DataLoader
from nlp import load_dataset
from utils import load_hotpot, load_complex, load_qangaroo, load_lama, load_search

"""
class Pretrain(Dataset):
    def __init__(self, tokenizer, type_path, num_samples, input_length, output_length, print_text=False, ssm=False):
        super().__init__()
       if ssm: 
           self.ssm = True
           self.nlp = pipeline('ner')
        else:
            self.ssm = False
        self.dataset = load_pretrain("./pretrain_sentence", split=type_path)
        self.input_length = input_length
        self.tokenizer = tokenizer
        self.output_length = output_length 
        self.print_text = print_text 

        self.sentinels = []
        for i in range(100):
            self.sentinels.append(f"<extra_id_{i}>")

    '''
    masking by SSM
    '''
    def salient_span_corruption_mask(self, text):

    '''
    original T5 span masking
    '''
    def span_corruption_mask(self, text):

    '''
    add sentinel to mask part of the text for pretrain input
    text:
    mask: 
    return:
    '''
    def input_span_to_unique_sentinel(self, text, mask):

    '''
    add sentinel to un-mask part of the text for pretrain output
    text:
    mask:
    return:
    '''
    def target_span_to_unique_sentinel(self, text, mask):

    def convert_to_features(self, example_batch):
        if self.print_text:
            print("Input Text: ", self.clean_text(example_batch['context']))
        text = self.clean_text(example_batch['context'])
        if self.ssm:
            mask = self.salient_span_corruption_mask(text)
        else:
            mask = self.span_corruption_mask(text)
        input_ = self.input_span_to_unique_sentinel(text, mask)
        target_ = self.target_span_to_unique_sentinel(text, mask)
        source = self.tokenizer.batch_encode_plus([input_], max_length=self.input_length, padding='max_length', truncation=True, return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([target_], max_length=self.input_length, padding='max_length', truncation=True, return_tensors='pt')
"""


class Trivia_QA_Closedbook(Dataset):
    def __init__(
        self,
        tokenizer,
        type_path,
        num_samples,
        input_length,
        output_length,
        print_text=False,
        add_all=False,
    ):
        super().__init__()
        self.dataset = load_dataset(
            "trivia_qa", "unfiltered.nocontext", split=type_path
        )
        if num_samples:
            rand_indices = np.random.choice(
                self.dataset.shape[0], num_samples, replace=False
            )
            self.dataset = self.dataset.select(list(rand_indices))
        self.input_length = input_length
        self.tokenizer = tokenizer
        self.output_length = output_length
        self.print_text = print_text

    def __len__(self):
        return self.dataset.shape[0]

    def clean_text(self, text):
        text = text.replace("Example of text:", "")
        text = text.replace("Example of Summary:", "")
        text = text.replace("\n", "")
        text = text.replace("``", "")
        text = text.replace('"', "")

        return text

    def convert_to_features(self, example_batch):
        # Tokenize contexts and questions (as pairs of inputs)

        if self.print_text:
            print("Input Text: ", self.clean_text(example_batch["question"]))
        #         input_ = self.clean_text(example_batch['text']) + " </s>"
        #         target_ = self.clean_text(example_batch['headline']) + " </s>"

        input_ = self.clean_text(example_batch["question"])
        target_ = self.clean_text(example_batch["answer"]["value"])

        source = self.tokenizer.batch_encode_plus(
            [input_],
            max_length=self.input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        targets = self.tokenizer.batch_encode_plus(
            [target_],
            max_length=self.output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return source, targets

    def __getitem__(self, index):
        source, targets = self.convert_to_features(self.dataset[index])

        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
        }


class Hotpot_QA_Closedbook(Dataset):
    def __init__(
        self,
        tokenizer,
        type_path,
        num_samples,
        input_length,
        output_length,
        print_text=False,
        add_all=False,
    ):
        self.dataset = load_hotpot(
            type_path
        )  # type_path = "train", "validation", "test"
        """
        if num_samples:
            rand_indices = np.random.choice(len(self.dataset), num_samples, replace=False)
            self.dataset = self.dataset.select(list(rand_indices))
        """
        self.input_length = input_length
        self.tokenizer = tokenizer
        self.output_length = output_length
        self.print_text = print_text

    def __len__(self):
        return len(self.dataset)

    def clean_text(self, text):
        # text = text.replace('"', '')
        return text

    def convert_to_features(self, example_batch):
        # tokenize contexts and questions (as pairs of inputs)
        if self.print_text:
            print(f"Input text: {self.clean_text[example_batch['question']]}")
        input_ = self.clean_text(example_batch["question"])
        target_ = self.clean_text(example_batch["answer"])
        source = self.tokenizer.batch_encode_plus(
            [input_],
            max_length=self.input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        targets = self.tokenizer.batch_encode_plus(
            [target_],
            max_length=self.output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return source, targets

    def __getitem__(self, idx):
        source, targets = self.convert_to_features(self.dataset[idx])
        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
            "question": "",
        }


class Complex_QA_Closedbook(Dataset):
    def __init__(
        self,
        tokenizer,
        type_path,
        num_samples,
        input_length,
        output_length,
        print_text=False,
        add_all=False,
    ):
        self.dataset = load_complex(
            type_path, add_all
        )  # type_path = "train", "validation", "test"
        """
        if num_samples:
            rand_indices = np.random.choice(len(self.dataset), num_samples, replace=False)
            self.dataset = self.dataset.select(list(rand_indices))
        """
        self.input_length = input_length
        self.tokenizer = tokenizer
        self.output_length = output_length
        self.print_text = print_text

    def __len__(self):
        return len(self.dataset)

    def clean_text(self, text):
        # text = text.replace('"', '')
        return text

    def convert_to_features(self, example_batch):
        # tokenize contexts and questions (as pairs of inputs)
        if self.print_text:
            print(f"Input text: {self.clean_text[example_batch['question']]}")
        input_ = self.clean_text(example_batch["question"])
        target_ = self.clean_text(example_batch["answer"])
        source = self.tokenizer.batch_encode_plus(
            [input_],
            max_length=self.input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        targets = self.tokenizer.batch_encode_plus(
            [target_],
            max_length=self.output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return source, targets

    def __getitem__(self, idx):
        source, targets = self.convert_to_features(self.dataset[idx])
        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
        }


"""
change!
"""


class Qangaroo_QA_Closedbook(Dataset):
    def __init__(
        self,
        tokenizer,
        type_path,
        num_samples,
        input_length,
        output_length,
        print_text=False,
        add_all=False,
    ):
        self.dataset = load_qangaroo(
            type_path, add_all
        )  # type_path = "train", "validation", "test"
        """
        if num_samples:
            rand_indices = np.random.choice(len(self.dataset), num_samples, replace=False)
            self.dataset = self.dataset.select(list(rand_indices))
        """
        self.input_length = input_length
        self.tokenizer = tokenizer
        self.output_length = output_length
        self.print_text = print_text

    def __len__(self):
        return len(self.dataset)

    def clean_text(self, text):
        # text = text.replace('"', '')
        return text

    def convert_to_features(self, example_batch):
        # tokenize contexts and questions (as pairs of inputs)
        if self.print_text:
            print(f"Input text: {self.clean_text[example_batch['question']]}")
        input_ = self.clean_text(example_batch["question"])
        target_ = self.clean_text(example_batch["answer"])
        source = self.tokenizer.batch_encode_plus(
            [input_],
            max_length=self.input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        targets = self.tokenizer.batch_encode_plus(
            [target_],
            max_length=self.output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return source, targets

    def __getitem__(self, idx):
        source, targets = self.convert_to_features(self.dataset[idx])
        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
            "question": question,
            "answer": answer,
        }


class LAMA_QA_Closedbook(Dataset):
    def __init__(
        self,
        tokenizer,
        type_path,
        num_samples,
        input_length,
        output_length,
        print_text=False,
        add_all=False,
    ):
        self.dataset = load_lama(
            type_path, add_all
        )  # type_path = "train", "validation", "test"
        """
        if num_samples:
            rand_indices = np.random.choice(len(self.dataset), num_samples, replace=False)
            self.dataset = self.dataset.select(list(rand_indices))
        """
        self.input_length = input_length
        self.tokenizer = tokenizer
        self.output_length = output_length
        self.print_text = print_text

    def __len__(self):
        return len(self.dataset)

    def clean_text(self, text):
        # text = text.replace('"', '')
        return text

    def convert_to_features(self, example_batch):
        # tokenize contexts and questions (as pairs of inputs)
        if self.print_text:
            print(f"Input text: {self.clean_text[example_batch['question']]}")
        input_ = self.clean_text(example_batch["question"])
        target_ = self.clean_text(example_batch["answer"])
        source = self.tokenizer.batch_encode_plus(
            [input_],
            max_length=self.input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        targets = self.tokenizer.batch_encode_plus(
            [target_],
            max_length=self.output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return source, targets, example_batch["question"], example_batch["answer"]

    def __getitem__(self, idx):
        source, targets, question, answer = self.convert_to_features(self.dataset[idx])
        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
            "question": question,
            "answer": answer,
        }


class Search_QA_Closedbook(Dataset):
    def __init__(
        self,
        tokenizer,
        type_path,
        num_samples,
        input_length,
        output_length,
        print_text=False,
        add_all=False,
    ):
        self.dataset = load_search(
            type_path, add_all
        )  # type_path = "train", "validation", "test"
        """
        if num_samples:
            rand_indices = np.random.choice(len(self.dataset), num_samples, replace=False)
            self.dataset = self.dataset.select(list(rand_indices))
        """
        self.input_length = input_length
        self.tokenizer = tokenizer
        self.output_length = output_length
        self.print_text = print_text

    def __len__(self):
        return len(self.dataset)

    def clean_text(self, text):
        # text = text.replace('"', '')
        return text

    def convert_to_features(self, example_batch):
        # tokenize contexts and questions (as pairs of inputs)
        if self.print_text:
            print(f"Input text: {self.clean_text[example_batch['question']]}")
        input_ = self.clean_text(example_batch["question"])
        target_ = self.clean_text(example_batch["answer"])
        source = self.tokenizer.batch_encode_plus(
            [input_],
            max_length=self.input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        targets = self.tokenizer.batch_encode_plus(
            [target_],
            max_length=self.output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return source, targets

    def __getitem__(self, idx):
        source, targets = self.convert_to_features(self.dataset[idx])

        question = self.dataset[idx]["question"]
        answer = self.dataset[idx]["answer"]

        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
            "question": question,
            "answer": answer,
        }


def get_dataset(tokenizer, type_path, num_samples, args):
    if args.dataset == "trivia":
        return Trivia_QA_Closedbook(
            tokenizer=tokenizer,
            type_path=type_path,
            num_samples=num_samples,
            input_length=args.max_input_length,
            output_length=args.max_output_length,
            add_all=args.add_all,
        )
    elif args.dataset == "hotpot":
        return Hotpot_QA_Closedbook(
            tokenizer=tokenizer,
            type_path=type_path,
            num_samples=num_samples,
            input_length=args.max_input_length,
            output_length=args.max_output_length,
            add_all=args.add_all,
        )
    elif args.dataset == "complex":
        return Complex_QA_Closedbook(
            tokenizer=tokenizer,
            type_path=type_path,
            num_samples=num_samples,
            input_length=args.max_input_length,
            output_length=args.max_output_length,
            add_all=args.add_all,
        )
    elif args.dataset == "qangaroo":
        return Qangaroo_QA_Closedbook(
            tokenizer=tokenizer,
            type_path=type_path,
            num_samples=num_samples,
            input_length=args.max_input_length,
            output_length=args.max_output_length,
            add_all=args.add_all,
        )
    elif args.dataset == "lama":
        return LAMA_QA_Closedbook(
            tokenizer=tokenizer,
            type_path=type_path,
            num_samples=num_samples,
            input_length=args.max_input_length,
            output_length=args.max_output_length,
            add_all=args.add_all,
        )
    elif args.dataset == "search":
        return Search_QA_Closedbook(
            tokenizer=tokenizer,
            type_path=type_path,
            num_samples=num_samples,
            input_length=args.max_input_length,
            output_length=args.max_output_length,
            add_all=args.add_all,
        )
    else:
        sys.exit()
