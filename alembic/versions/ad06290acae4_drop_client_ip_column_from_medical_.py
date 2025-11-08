"""drop client_ip column from medical_predictions

Revision ID: ad06290acae4
Revises: 5d6e0a745cf2
Create Date: 2025-11-05 16:59:18.767333

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ad06290acae4'
down_revision: Union[str, Sequence[str], None] = '5d6e0a745cf2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_column('medical_predictions', 'client_ip')


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column('medical_predictions', sa.Column('client_ip', sa.String(), nullable=True))
