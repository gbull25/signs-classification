"""Database revision

Revision ID: abbb05bcd801
Revises: 485da13da336
Create Date: 2024-03-16 12:11:18.577527

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'abbb05bcd801'
down_revision = '485da13da336'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('role',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('permissions', sa.JSON(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.add_column('user', sa.Column('is_verified', sa.Boolean(), nullable=False))
    op.alter_column('user', 'registered_at',
               existing_type=postgresql.TIMESTAMP(),
               nullable=True)
    op.alter_column('user', 'role_id',
               existing_type=sa.INTEGER(),
               nullable=True)
    op.create_foreign_key(None, 'user', 'role', ['role_id'], ['id'])
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint(None, 'user', type_='foreignkey')
    op.alter_column('user', 'role_id',
               existing_type=sa.INTEGER(),
               nullable=False)
    op.alter_column('user', 'registered_at',
               existing_type=postgresql.TIMESTAMP(),
               nullable=False)
    op.drop_column('user', 'is_verified')
    op.drop_table('role')
    # ### end Alembic commands ###