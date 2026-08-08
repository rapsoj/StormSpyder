import argparse

from stormspyder.pipeline import run_pipeline_with_email


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Run the StormSpyder forecasting pipeline')
    parser.add_argument(
        '--local-testing',
        action='store_true',
        help='Use a non-headless local Chrome setup for testing instead of the default production browser configuration.',
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_pipeline_with_email(local_testing=args.local_testing)


if __name__ == '__main__':
    main()
