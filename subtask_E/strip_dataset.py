from csv import DictWriter
from argparse import ArgumentParser
from lxml import etree
from os import path
from os.path import getsize

ORG_QUESTION_ID = 'original_question_id'
REL_QUESTION_ID = 'related_question_id'
ORG_QUESTION_TEXT = 'original_question_text'
REL_QUESTION_TEXT = 'related_question_text'


class DatasetIterator(object):
    def __init__(self, fp):
        self.fp = fp

    def _iter_cleanup(self, element):
        element.clear()

        while element.getprevious() is not None:
            del element.getparent()[0]

    def __iter__(self):
        for event, original_question in etree.iterparse(self.fp, tag='OrgQuestion'):
            original_question_id = original_question.get('ORGQ_ID')
            original_question_text = original_question.findtext("OrgQBody")

            related_question = original_question.find(".//RelQuestion")
            related_question_id = related_question.get('RELQ_ID')
            related_question_text = related_question.findtext('RelQBody')

            yield { ORG_QUESTION_ID: original_question_id,
                    REL_QUESTION_ID: related_question_id,
                    ORG_QUESTION_TEXT: original_question_text,
                    REL_QUESTION_TEXT: related_question_text }

            self._iter_cleanup(original_question)


def strip_dataset(xml_path, output, verbose):
    fieldnames = [ORG_QUESTION_ID, REL_QUESTION_ID,
                  ORG_QUESTION_TEXT, REL_QUESTION_TEXT]

    with open(output, 'w') as csv_fp:
        writer = DictWriter(csv_fp, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()

        with open(xml_path, 'rb') as fp:
            total_xml_size = getsize(xml_path)

            for row in DatasetIterator(fp):
                writer.writerow(row)

                if verbose:
                    progress = float(fp.tell() / total_xml_size) * 100.0
                    print('File progress = {:2.2f}%'.format(progress), end='\r')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', dest='data', help='Input dataset .xml path')
    parser.add_argument('--output', dest='output', help='Output .csv path')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    strip_dataset(args.data, args.output, args.verbose)
