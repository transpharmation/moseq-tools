import os
import yaml
import argparse


TEMPLATE_CONFIG_FILE = '/mnt/DATA1_8TB/templates_moseq/minimal_config_combined.yaml'
TEMPLATE_ROOT = 'moseq-code'
TARGET_CONFIG_FILE = 'config.yaml'

class IndentDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(IndentDumper, self).increase_indent(flow, False)

def replace_path_component(path, template, replacement):
    """ Replace <template> path component with <replacement> path component """
    # Split the path into components
    path_components = path.split(os.sep)
    # Replace the template component with the replacement
    new_path_components = [
        os.path.abspath(replacement) if component == template 
        else component for component in path_components
    ]
    # Join the components back into a path
    new_path = os.sep.join(new_path_components).replace(os.sep*2, os.sep)
    return new_path


def apply_path_to_config(
        path, config_path=TEMPLATE_CONFIG_FILE, template=TEMPLATE_ROOT
        ):
    """ Apply <path> to all values where the template path is found in the YAML 
        configuration file.
    """
    content = open(config_path).read()
    data = yaml.safe_load(content)
    for key, value in data.items():
        # Find configuration values that have <TEMPLATE_PATH> within
        if isinstance(value, str) and template in value:
            data[key] = replace_path_component(value, template, path)

    with open(TARGET_CONFIG_FILE, 'w') as f:
        f.write(
            yaml.dump(data, Dumper=IndentDumper)
        )

def apply_path():
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--template', type=str, default=TEMPLATE_CONFIG_FILE, help='Path to the template configuration file')
    parser.add_argument('--template-root', type=str, default=TEMPLATE_ROOT, help='Override the default template root value')
    parser.add_argument('--target', type=str, default=TARGET_CONFIG_FILE, help='Override the default target configuration file value')
    parser.add_argument('path', type=str, default=os.getcwd(), nargs='?')

    args = parser.parse_args()

    path = args.path

    if not os.path.exists(path):
        print("Invalid study path specified")
        sys.exit(-1)
    if not os.path.exists(args.template):
        print(f"Template configuration file, {args.template} not found.")
        sys.exit(-1)

    apply_path_to_config(path=path, config_path=args.template, template=args.template_root)

    with open(args.target, 'w') as f:
        f.write(
            yaml.dump(yaml.safe_load(open(args.template).read()), Dumper=IndentDumper)
        )


if __name__ == '__main__':
    apply_path()