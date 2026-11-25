import unittest

from scripts.run_hygon_tf_ttcl_parallel_experimental import validate_idle_devices, validate_urls


class HygonParallelLauncherTests(unittest.TestCase):
    def test_requires_seven_distinct_loopback_services(self):
        urls = [f'http://127.0.0.1:{19500 + i}/v1' for i in range(1, 8)]
        self.assertEqual(validate_urls(urls), urls)
        with self.assertRaises(ValueError):
            validate_urls(urls[:6] + [urls[0]])
        with self.assertRaises(ValueError):
            validate_urls(urls[:6] + ['http://10.0.0.1:19507/v1'])

    def test_hcu_zero_can_be_busy_but_every_other_hcu_must_be_idle(self):
        def report(busy=None, missing=None):
            return '\n'.join(
                f'HCU[{index}] : {kind}: {9 if index == busy or index == 0 else 0}'
                for kind in ('HCU use (%)', 'HCU memory use (%)')
                for index in range(8) if index != missing)
        validate_idle_devices(report())
        with self.assertRaisesRegex(ValueError, 'not all idle'):
            validate_idle_devices(report(busy=4))
        with self.assertRaisesRegex(ValueError, 'not all idle'):
            validate_idle_devices(report(missing=2))


if __name__ == '__main__':
    unittest.main()
