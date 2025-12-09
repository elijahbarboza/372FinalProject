"""
Utility functions for loading and managing player data.
"""
import pandas as pd
import os


def load_player_data(csv_path):
    """
    Load player data from CSV file.
    
    Args:
        csv_path: Path to players.csv file
        
    Returns:
        DataFrame with player information
    """
    return pd.read_csv(csv_path)


def get_player_name(player_id, player_df):
    """
    Get player full name from player ID.
    
    Args:
        player_id: Player ID (as string or int)
        player_df: DataFrame with player data
        
    Returns:
        Full name string (e.g., "Chris Paul") or None if not found
    """
    player_id = str(player_id)
    player_row = player_df[player_df['playerid'] == player_id]
    if len(player_row) > 0:
        fname = player_row.iloc[0]['fname']
        lname = player_row.iloc[0]['lname']
        return f"{fname} {lname}"
    return None


def get_player_info(player_id, player_df):
    """
    Get full player information from player ID.
    
    Args:
        player_id: Player ID (as string or int)
        player_df: DataFrame with player data
        
    Returns:
        Dictionary with player information or None if not found
    """
    player_id = str(player_id)
    player_row = player_df[player_df['playerid'] == player_id]
    if len(player_row) > 0:
        return player_row.iloc[0].to_dict()
    return None


def create_player_id_to_name_map(csv_path):
    """
    Create a mapping from player ID to full name.
    
    Args:
        csv_path: Path to players.csv file
        
    Returns:
        Dictionary mapping player_id (string) to full name
    """
    df = load_player_data(csv_path)
    mapping = {}
    for _, row in df.iterrows():
        player_id = str(row['playerid'])
        full_name = f"{row['fname']} {row['lname']}"
        mapping[player_id] = full_name
    return mapping

