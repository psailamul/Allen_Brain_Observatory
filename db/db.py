#!/usr/bin/env python
import sys
import sshtunnel
import argparse
import psycopg2
import psycopg2.extras
import psycopg2.extensions
import credentials
import config
sshtunnel.DAEMON = True  # Prevent hanging process due to forward thread
main_config = config.Allen_Brain_Observatory_Config()


class db(object):
    def __init__(self, config):
        self.status_message = False
        self.db_schema_file = 'db/db_schema.txt'
        # Pass config -> this class
        for k, v in config.items():
            setattr(self, k, v)

    def __enter__(self):
        if main_config.db_ssh_forward:
            forward = sshtunnel.SSHTunnelForwarder(
                credentials.machine_credentials()['ssh_address'],
                ssh_username=credentials.machine_credentials()['username'],
                ssh_password=credentials.machine_credentials()['password'],
                remote_bind_address=('127.0.0.1', 5432))
            forward.start()
            self.forward = forward
            self.pgsql_port = forward.local_bind_port
        else:
            self.forward = None
            self.pgsql_port = ''
        pgsql_string = credentials.postgresql_connection(str(self.pgsql_port))
        self.pgsql_string = pgsql_string
        self.conn = psycopg2.connect(**pgsql_string)
        self.conn.set_isolation_level(
            psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        self.cur = self.conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            print exc_type, exc_value, traceback
            self.close_db(commit=False)
        else:
            self.close_db()
        if main_config.db_ssh_forward:
            self.forward.close()
        return self

    def close_db(self, commit=True):
        self.conn.commit()
        self.cur.close()
        self.conn.close()

    def return_status(
            self,
            label,
            throw_error=False):
        """
        General error handling and status of operations.
        ::
        label: a string of the SQL operation (e.g. 'INSERT').
        throw_error: if you'd like to terminate execution if an error.
        """
        if label in self.cur.statusmessage:
            print 'Successful %s.' % label
        else:
            if throw_error:
                raise RuntimeError('%s' % self.cur.statusmessag)
            else:
                'Encountered error during %s: %s.' % (
                    label, self.cur.statusmessage
                    )

    def recreate_db(self, run=False):
        """Initialize the DB with db_schema_file."""
        if run:
            db_schema = open(self.db_schema_file).read().splitlines()
            for s in db_schema:
                t = s.strip()
                if len(t):
                    self.cur.execute(t)

    def populate_db_with_cell_stim(self, namedict):
        """
        Add cell stim info to the db.
        ::
        experiment_name: name of experiment to add
        parent_experiment: linking a child (e.g. clickme) -> parent (ILSVRC12)
        """
        self.cur.executemany(
            """
            INSERT INTO cells
            (cell_id, cell_npy, session, drifting_gratings, locally_sparse_noise, locally_sparse_noise_four_deg, locally_sparse_noise_eight_deg, natural_scenes, natural_movie_one, natural_movie_two, natural_movie_three, spontaneous, static_gratings)
            VALUES
            (%(cell_id)s, %(cell_npy)s, %(session)s, %(drifting_gratings)s, %(locally_sparse_noise)s, %(locally_sparse_noise_four_deg)s, %(locally_sparse_noise_eight_deg)s, %(natural_scenes)s, %(natural_movie_one)s, %(natural_movie_two)s, %(natural_movie_three)s, %(spontaneous)s, %(static_gratings)s)
            """,
            namedict)
        if self.status_message:
            self.return_status('INSERT')

    def populate_db_with_rf(self, namedict):
        """
        Add cell RF info to the db.
        ::
        experiment_name: name of experiment to add
        parent_experiment: linking a child (e.g. clickme) -> parent (ILSVRC12)
        """
        self.cur.executemany(
            """
            INSERT INTO cells
            (cell_id , lsn_name , alpha , number_of_shuffles, on_height , on_center_x , on_center_y , on_width_x , on_width_y , on_rotation , off_height , off_center_x , off_center_y , off_width_x , off_width_y , off_rotation , overlap_area , area )
            VALUES
            (%(cell_id)s, %(lsn_name)s, %(alpha)s, %(number_of_shuffles)s, %(on_height)s, %(on_center_x)s, %(on_center_y)s, %(on_width_x)s, %(on_width_y)s, %(on_rotation)s, %(off_height)s, %(off_center_x)s, %(off_center_y)s, %(off_width_x)s, %(off_width_y)s, %(off_rotation)s, %(overlap_area)s, %(area)s)
            """,
            namedict)
        if self.status_message:
            self.return_status('INSERT')



def initialize_database():
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        db_conn.recreate_db(run=True)
        db_conn.return_status('CREATE')


def add_cell_data(
        cell_rf_dict,
        list_of_cell_stim_dicts):
    """Add a cell to the databse.

    Inputs:::
    cell_rf_dict: dictionary containing cell_id_number and its RF properties.
    list_of_cell_stim_dicts: a list of dictionaries, each containing the cell's
        id + a pointer to a data numpy file and a boolean for the stimuli it contains.
    ------------------------------
    For a given cell, e.g., cell_1

    cell_rf_dict = {
        'cell_id': 1,
        'rf': big
    }
    list_of_cell_stim_dicts[
        {
            'cell_id': 1,
            'session': A,
            'drifting_gratings': True,
            'ALL OTHER COLUMNS': False,
            'cell_npy': os.path.join(data_dir, '%s_%s_%s.npy' % (cell_id, session, stimulus))
        },
        {
            'cell_id': 1,
            'session': B,
            'drifting_gratings': True,
            'ALL OTHER COLUMNS': False,
            'cell_npy': os.path.join(data_dir, '%s_%s_%s.npy' % (cell_id, session, stimulus))
        }
    ]    
    """
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        db_conn.populate_db_with_rf(cell_rf_dict)
        db_conn.populate_db_with_cell_stim(list_of_cell_stim_dicts)


def get_performance(experiment_name):
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        perf = db_conn.get_performance(experiment_name=experiment_name)
    return perf


def main(
        initialize_db,
        reset_process=False):
    if reset_process:
        reset_in_process()
    if initialize_db:
        print 'Initializing database.'
        initialize_database()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initialize",
        dest="initialize_db",
        action='store_true',
        help='Recreate your database.')
    args = parser.parse_args()
    main(**vars(args))
