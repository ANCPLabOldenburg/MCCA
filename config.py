import os
import yaml
import subprocess


def get_git_revision_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except subprocess.CalledProcessError as e:
        print(f"WARNING: Could not parse current git revision hash:\n{e}")
        return None


class Config:
    def __init__(self):
        self.results_folder = None
        self.n_ccs = None
        self.n_pcs = None
        with open('config.yaml', 'r') as file:
            self.__dict__.update(yaml.full_load(file))

        if self.n_ccs > self.n_pcs:
            import warnings
            warnings.warn(f"Warning........... number of MCCA components ({self.n_ccs}) cannot be greater "
                          f"than number of PCA components ({self.n_pcs}), setting them equal.")
            self.n_ccs = self.n_pcs

    def save(self):
        """ Update the config file with the commit hash and save it in its sub folder. """
        with open('config.yaml', 'r') as file:
            lines = file.readlines()

        # Add or update commit hash
        commit_hash = get_git_revision_hash()
        if commit_hash:
            # Find commit hash in config file
            idx = -1
            for i, line in enumerate(lines):
                if line.startswith("commit_hash: "):
                    idx = i
                    break

            if idx == -1:
                # If config file has no commit hash, just append it
                lines.append(f"commit_hash: {commit_hash} # The commit hash of this repo when this config was run\n")
            else:
                # If config file has a commit hash, warn if it is different from the current revision
                line = lines[idx].split()
                old_commit_hash = line[1]
                if old_commit_hash != commit_hash:
                    print(f"WARNING: this config file was originally run with revision {old_commit_hash}, "
                          f"current revision is {commit_hash}")
                # Update config file to the new commit hash
                line[1] = commit_hash
                lines[idx] = " ".join(line) + "\n"

        os.makedirs(self.results_folder, exist_ok=True)

        with open(self.results_folder + 'config.yaml', 'w') as file:
            file.write("".join(lines))


CONFIG = Config()
CONFIG.save()
