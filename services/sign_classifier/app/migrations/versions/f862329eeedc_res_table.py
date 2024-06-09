"""res_table

Revision ID: f862329eeedc
Revises: abbb05bcd801
Create Date: 2024-06-09 17:04:39.541776

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'f862329eeedc'
down_revision = 'abbb05bcd801'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('results',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.String(), nullable=False),
    sa.Column('result_filepath', sa.String(), nullable=False),
    sa.Column('frame_num', sa.Integer(), nullable=True),
    sa.Column('detection_id', sa.Integer(), nullable=False),
    sa.Column('detection_conf', sa.Float(), nullable=False),
    sa.Column('sign_class', sa.Integer(), nullable=False),
    sa.Column('sign_description', sa.String(), nullable=False),
    sa.Column('bbox', sa.String(), nullable=False),
    sa.Column('frame_number', sa.Integer(), nullable=False),
    sa.Column('detection_speed', sa.Integer(), nullable=False),
    sa.Column('model_used', sa.String(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    # op.create_foreign_key(None, 'results', 'rating', ['user_id'], ['user_id'])
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('results')
    # ### end Alembic commands ###