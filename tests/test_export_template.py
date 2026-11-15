"""Literal safety for generated-source indentation compaction."""
import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location(
    "export_template", Path(__file__).resolve().parents[1] / "scripts/export-template.py")
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


class IndentationTests(unittest.TestCase):
    def test_tight_packing_keeps_literals_directives_and_operators(self):
        source = ('#define OBJECT (1 * 2)\n'
                  'auto s = R"( * & % [ ] = literal )";\n'
                  'auto t = " a * b ";\n'
                  'auto x = a * b; auto p = &x;\n')
        actual = export.compact_indentation(source, tight=True)
        self.assertIn('#define OBJECT (1 * 2)\n', actual)
        self.assertIn('R"( * & % [ ] = literal )"', actual)
        self.assertIn('" a * b "', actual)
        self.assertIn('auto x=a*b;auto p=&x;', actual)
        for operators in ('& &', '| |', '* /', '/ *', '* =', '% :', '< %', '= =',
                          '+ +', '- -', '- >', '+ =', '- =', '1e +', '0x1p -',
                          '< <', '> >', '< =', '> =', '! =', ': :', '< :', ': >',
                          '/ /', '/ ='):
            with self.subTest(operators=operators):
                packed = export.compact_indentation(operators, tight=True)
                self.assertNotEqual(packed, operators.replace(' ', ''))
        spliced = '#define X a\\\n b\n'
        self.assertEqual(export.compact_indentation(spliced, tight=True), spliced)

    def test_line_splicing_cannot_join_tokens(self):
        for ending in ('\n', '\r\n'):
            source = '  #define X a\\' + ending + ' b' + ending
            actual = export.compact_indentation(source)
            self.assertEqual(actual, source)
            self.assertIn('a b', actual.replace('\\' + ending, ''))
        source = '  /* *\\\n / text */\n  int x;\n'
        self.assertEqual(export.compact_indentation(source),
                         source)
        source = '/* *\\\n/\nauto s=R"(x\n    literal data\n)";\n'
        self.assertEqual(export.compact_indentation(source), source)
        source = '  // comment ??/\n    still a comment\n'
        self.assertEqual(export.compact_indentation(source), source)

    def test_standalone_comments_preserve_tokens_and_lines(self):
        source = '  // unmatched " R"( text\n  int x = __LINE__; // inline removed\n// EOF'
        self.assertEqual(export.compact_indentation(source),
                         '\nint x = __LINE__;\n')
        source = '/* block\n  // block text\n*/\n  // removed\nint y;\n'
        self.assertEqual(export.compact_indentation(source),
                         '/* block\n// block text\n*/\n\nint y;\n')
        source = 'auto s=R"tag(\n  // literal text\n)tag";\n // removed\n'
        self.assertEqual(export.compact_indentation(source),
                         'auto s=R"tag(\n  // literal text\n)tag";\n\n')

    def test_inline_comments_preserve_literal_and_macro_boundaries(self):
        source = ('#define K 7 // trailing comment\n'
                  'auto u = "https://example.test"; // URL is literal\n'
                  'int n // not part of the identifier\n'
                  '= __LINE__; // unmatched " R"(\n')
        self.assertEqual(export.compact_indentation(source),
                         '#define K 7 \nauto u = "https://example.test";\n'
                         'int n \n= __LINE__;\n')

    def test_closing_punctuation_keeps_separate_tokens(self):
        source = 'if (f()) { g(); } else { h(); }\n'
        self.assertEqual(export.compact_indentation(source),
                         'if (f()){ g();}else { h();}\n')

    def test_literal_contents_and_line_count(self):
        cases = [
            ('    const char *s = "  x  ";\n', 'const char *s = "  x  ";\n'),
            ('  auto s = R"(first\n    literal data\n  last)";\n  int x;\n',
             'auto s = R"(first\n    literal data\n  last)";\nint x;\n'),
            ('  auto s = u8R"tag(\n    // \" quote\n  )tag";\n  int x;\n',
             'auto s = u8R"tag(\n    // \" quote\n  )tag";\nint x;\n'),
            ('  auto s = "abc\\\n    literal data";\n  int x;\n',
             '  auto s = "abc\\\n    literal data";\n  int x;\n'),
            ('  char q = \'\\\'\';\n  auto s = "\\\" // text";\n',
             'char q = \'\\\'\';\nauto s = "\\\" // text";\n'),
            ('  // quote \" \\\n    still comment R"(\n  int x;\n',
             '  // quote \" \\\n    still comment R"(\n  int x;\n'),
            ('  /* R"( \" comment\n    end */\n  int x;\n',
             '/* R"( \" comment\nend */\nint x;\n'),
            ("  auto n = 1'234;\n  auto m = 0xF'F;\n", "auto n = 1'234;\nauto m = 0xF'F;\n"),
            ('  #define X(a) \\\n    ((a) + 1)\n', '  #define X(a) \\\n    ((a) + 1)\n'),
        ]
        for source, expected in cases:
            with self.subTest(source=source):
                actual = export.compact_indentation(source)
                self.assertEqual(actual, expected)
                self.assertEqual(actual.count('\n'), source.count('\n'))


if __name__ == "__main__":
    unittest.main()
