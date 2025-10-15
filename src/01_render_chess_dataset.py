import chess
import chess.pgn
import chess.svg
import cairosvg
import requests
from pprint import pprint
import io
import os
import json
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np
import random
from sklearn.model_selection import train_test_split

from PIL import Image, ImageDraw, ImageFilter

# Dataset directorie
current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = pathlib.Path(current_dir).parent
DATASET_NAME = "render_chess_dataset"
DATASET_DIR = ROOT_DIR / "data" / DATASET_NAME

# Pieces images directory
PIECES_DIR = ROOT_DIR / "data" / "piece_images"
PIECES_DIR.mkdir(exist_ok=True)

N_CHESS_GAMES = 25
TEST_TRAIN_RATIO = 0.2
VAL_TEST_RATIO = 0.1

TOURNAMENT_URL_IDS = ["late-titled-tuesday-blitz-may-13-2025-5643227"]

HTTP_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "chess-api-test/0.1 (username: AlexCQJ; contact: alex.c.quiroga.jaldin@gmail.com)",
}

# Default chess.com color layout
# BOARD_COLORS = ("#EBECD3", "#7D945D")

# Multiple color schemes
BOARD_COLORS_LIST = [
    ("#EBECD3", "#7D945D"),  # Chess com (estandar)
    ("#EFF788", "#BAD350"),  # Chess com (move)
    ("#B98F4E", "#805531"),
    ("#6A7383", "#292E3B"),
    ("#EDD6B0", "#B88762"),
    ("#D8E3E7", "#789BB0"),
    ("#F5DBC3", "#BB5746"),
    ("#E5E3DE", "#306448"),
]

CHESS_PIECES_PATHS = [
    {
        "K": PIECES_DIR / "chess_com" / "wK.png",
        "Q": PIECES_DIR / "chess_com" / "wQ.png",
        "R": PIECES_DIR / "chess_com" / "wR.png",
        "B": PIECES_DIR / "chess_com" / "wB.png",
        "N": PIECES_DIR / "chess_com" / "wN.png",
        "P": PIECES_DIR / "chess_com" / "wP.png",
        "k": PIECES_DIR / "chess_com" / "bK.png",
        "q": PIECES_DIR / "chess_com" / "bQ.png",
        "r": PIECES_DIR / "chess_com" / "bR.png",
        "b": PIECES_DIR / "chess_com" / "bB.png",
        "n": PIECES_DIR / "chess_com" / "bN.png",
        "p": PIECES_DIR / "chess_com" / "bP.png",
    },
    {
        "K": PIECES_DIR / "chess_com_2" / "wK.png",
        "Q": PIECES_DIR / "chess_com_2" / "wQ.png",
        "R": PIECES_DIR / "chess_com_2" / "wR.png",
        "B": PIECES_DIR / "chess_com_2" / "wB.png",
        "N": PIECES_DIR / "chess_com_2" / "wN.png",
        "P": PIECES_DIR / "chess_com_2" / "wP.png",
        "k": PIECES_DIR / "chess_com_2" / "bK.png",
        "q": PIECES_DIR / "chess_com_2" / "bQ.png",
        "r": PIECES_DIR / "chess_com_2" / "bR.png",
        "b": PIECES_DIR / "chess_com_2" / "bB.png",
        "n": PIECES_DIR / "chess_com_2" / "bN.png",
        "p": PIECES_DIR / "chess_com_2" / "bP.png",
    },
    {
        "K": PIECES_DIR / "chess_com_3" / "wK.png",
        "Q": PIECES_DIR / "chess_com_3" / "wQ.png",
        "R": PIECES_DIR / "chess_com_3" / "wR.png",
        "B": PIECES_DIR / "chess_com_3" / "wB.png",
        "N": PIECES_DIR / "chess_com_3" / "wN.png",
        "P": PIECES_DIR / "chess_com_3" / "wP.png",
        "k": PIECES_DIR / "chess_com_3" / "bK.png",
        "q": PIECES_DIR / "chess_com_3" / "bQ.png",
        "r": PIECES_DIR / "chess_com_3" / "bR.png",
        "b": PIECES_DIR / "chess_com_3" / "bB.png",
        "n": PIECES_DIR / "chess_com_3" / "bN.png",
        "p": PIECES_DIR / "chess_com_3" / "bP.png",
    },
    {
        "K": PIECES_DIR / "chess_com_4" / "wK.png",
        "Q": PIECES_DIR / "chess_com_4" / "wQ.png",
        "R": PIECES_DIR / "chess_com_4" / "wR.png",
        "B": PIECES_DIR / "chess_com_4" / "wB.png",
        "N": PIECES_DIR / "chess_com_4" / "wN.png",
        "P": PIECES_DIR / "chess_com_4" / "wP.png",
        "k": PIECES_DIR / "chess_com_4" / "bK.png",
        "q": PIECES_DIR / "chess_com_4" / "bQ.png",
        "r": PIECES_DIR / "chess_com_4" / "bR.png",
        "b": PIECES_DIR / "chess_com_4" / "bB.png",
        "n": PIECES_DIR / "chess_com_4" / "bN.png",
        "p": PIECES_DIR / "chess_com_4" / "bP.png",
    },
    {
        "K": PIECES_DIR / "wikipedia" / "wK.png",
        "Q": PIECES_DIR / "wikipedia" / "wQ.png",
        "R": PIECES_DIR / "wikipedia" / "wR.png",
        "B": PIECES_DIR / "wikipedia" / "wB.png",
        "N": PIECES_DIR / "wikipedia" / "wN.png",
        "P": PIECES_DIR / "wikipedia" / "wP.png",
        "k": PIECES_DIR / "wikipedia" / "bK.png",
        "q": PIECES_DIR / "wikipedia" / "bQ.png",
        "r": PIECES_DIR / "wikipedia" / "bR.png",
        "b": PIECES_DIR / "wikipedia" / "bB.png",
        "n": PIECES_DIR / "wikipedia" / "bN.png",
        "p": PIECES_DIR / "wikipedia" / "bP.png",
    },
]


def get_player_games(username="Hikaru", year="2025", month="01"):
    url = f"https://api.chess.com/pub/player/{username}/games/{year}/{month}"
    response = requests.get(url, headers=HTTP_HEADERS)
    if response.status_code == 200:
        player_games_archive = response.json()
        print("Keys: ", player_games_archive.keys())
        print("games keys: ", player_games_archive["games"][0].keys())
        print("Number of games: ", len(player_games_archive["games"]))
        selected_games = player_games_archive["games"]
        # pprint(selected_games[0])
        return selected_games
    else:
        print(
            f"Error fetching games archive for {username} in {year}-{month}: {response.status_code}"
        )
    return []


def get_tournament_rounds(tournament_url_id):
    url = f"https://api.chess.com/pub/tournament/{tournament_url_id}"
    response = requests.get(url, headers=HTTP_HEADERS)
    if response.status_code == 200:
        tournament_details = response.json()
        print("==== Tournament Details ====")
        print("Keys: ", tournament_details.keys())
        print("Name: ", tournament_details["name"])
        print("Number of players: ", len(tournament_details["players"]))
        print("Number of rounds: ", len(tournament_details["rounds"]))
        rounds_urls = tournament_details["rounds"]
        return rounds_urls
    else:
        print(f"Error fetching tournament details for {url_id}: {response.status_code}")
        return []


def get_tournament_round_groups(round_urls):
    groups_urls = []

    for round_url in tqdm(round_urls):
        response = requests.get(round_url, headers=HTTP_HEADERS)
        if response.status_code == 200:
            round_details = response.json()
            groups_urls += round_details.get("groups", [])
            print(f"Number of groups in round {round_url}: {len(groups_urls)}")
        else:
            print(
                f"Error fetching round details for {round_url}: {response.status_code}"
            )
    return groups_urls


def get_games_in_rounds(round_urls):
    games = []

    for round_url in tqdm(round_urls):
        response = requests.get(round_url, headers=HTTP_HEADERS)
        if response.status_code == 200:
            round_details = response.json()
            games += round_details.get("games", [])
        else:
            print(
                f"Error fetching round details for {round_url}: {response.status_code}"
            )

    print(f"Total number of games in all rounds: {len(games)}")
    return games


def get_game_info(game_number):
    game_json = selected_games[game_number]
    game_pgn = game_json["pgn"]
    player_white = game_json["white"]
    player_black = game_json["black"]

    return {"pgn": game_pgn, "white": player_white, "black": player_black}


def board2arr(board):
    pieces_map = board.piece_map()
    pieces_map = {k: v.symbol() for k, v in pieces_map.items()}

    PIECES_NUMBERS = [
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    ]

    pieces_squares = list(map(lambda x: pieces_map.get(x, "0"), PIECES_NUMBERS))

    pieces_arr = np.array(pieces_squares).reshape(8, 8)

    return pieces_arr


def get_board_representation(board):
    """Converts a chess.Board object to a 2D list representation."""
    board_str = str(board).replace(" ", "")
    rows = board_str.split("\n")
    board_repr = [list(row) for row in rows]
    return board_repr


def draw_board(board_repr, piece_paths, board_colors, player_pov="white", size=400):
    """
    Draws a chessboard with pieces loaded from image files.

    Args:
        board_repr (list): 2D list representing the board state.
        piece_paths (dict): Dictionary mapping piece symbols to image file paths.
        board_colors (tuple | List[tuple]): A tuple or list of tuples representing (light_square_color, dark_square_color).
        player_pov (str): The perspective of the board ('white' or 'black').
        size (int): The size of the board image in pixels.
    """
    if player_pov == "black":
        board_repr = board_repr[::-1]
        board_repr = [row[::-1] for row in board_repr]

    # Set board colors
    if isinstance(board_colors, list):
        board_colors = random.choice(board_colors)  # Choose a random color scheme

    if isinstance(piece_paths, list):
        piece_paths = random.choice(piece_paths)  # Choose a random set of piece paths

    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)

    square_size = size // 8
    light_color, dark_color = board_colors

    # Draw squares
    for row in range(8):
        for col in range(8):
            x1, y1 = col * square_size, row * square_size
            x2, y2 = x1 + square_size, y1 + square_size
            color = light_color if (row + col) % 2 == 0 else dark_color
            draw.rectangle([x1, y1, x2, y2], fill=color)

    # Place pieces
    for row in range(8):
        for col in range(8):
            piece_symbol = board_repr[row][col]
            if piece_symbol != "." and piece_symbol in piece_paths:
                # Open the piece image from the file path
                piece_img = Image.open(piece_paths[piece_symbol])

                # Resize piece to fit the square
                piece_img = piece_img.resize(
                    (square_size, square_size), Image.Resampling.LANCZOS
                )

                x, y = col * square_size, row * square_size

                # Paste the piece, using its alpha channel for transparency
                if piece_img.mode == "RGBA":
                    img.paste(piece_img, (x, y), piece_img)
                else:
                    img.paste(piece_img, (x, y))

    return img


def process_game(game_info, imgs_dir, labels_dir, piece_paths, board_colors):

    # Settings
    player_pov = "white"

    # Instantiate a chess game
    pgn = io.StringIO(game_info["pgn"])
    chess_game = chess.pgn.read_game(pgn)
    board = chess_game.board()

    # --- Render initial position (before any moves) ---
    board_array = get_board_representation(board)
    img_board = draw_board(
        board_array, piece_paths, board_colors, player_pov=player_pov
    )
    name = f"{game_info['white']['username']}_{game_info['black']['username']}_000"
    img_path = os.path.join(imgs_dir, name + ".png")
    img_board.save(img_path, "PNG")
    pieces_arr = board2arr(board)
    np.savetxt(
        os.path.join(labels_dir, f"{name}.csv"), pieces_arr, delimiter=",", fmt="%s"
    )

    # Iterate through moves and generate images and labels
    for i, move in enumerate(list(chess_game.mainline_moves())):
        # print(f"Move {i + 1}: {move}")
        board.push(move)

        # Get the 2D list representation of the board
        board_array = get_board_representation(board)

        # Render the board with your custom PNG pieces
        img_board = draw_board(
            board_array, piece_paths, board_colors, player_pov=player_pov
        )

        # Apply a level of blur randomly
        # if random.random() < 0.5:  # 50% chance to apply blur
        #     img_board = img_board.filter(ImageFilter.GaussianBlur(radius=2))

        radius = random.randint(0, 2)
        img_board = img_board.filter(ImageFilter.GaussianBlur(radius=radius))

        # Save as PNG
        name = f"{game_info['white']['username']}_{game_info['black']['username']}_{i + 1:03d}"
        img_path = os.path.join(imgs_dir, name + ".png")
        img_board.save(img_path, "PNG")

        # Generate fen and save labels
        # fen = board.fen()

        # Get arrays of pieces
        pieces_arr = board2arr(board)
        # Save pieces array to CSV
        np.savetxt(
            os.path.join(labels_dir, f"{name}.csv"), pieces_arr, delimiter=",", fmt="%s"
        )


def create_split_directory(split_dir):
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)

    img_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    return img_dir, labels_dir


def create_labels_mapping(save_dir):

    board = chess.Board()

    pieces_arr = board2arr(board)

    class_names = sorted(list(set(pieces_arr.flatten())))

    class2id = {name: id for id, name in enumerate(class_names)}
    id2class = {id: name for name, id in class2id.items()}

    map_path = save_dir / "labels_mapping.json"
    with open(map_path, "w") as f:
        json.dump({"class2id": class2id, "id2class": id2class}, f, indent=4)


def process_games(train_games, test_games, validation_games):

    # Pieces images paths

    # Train games
    train_dir = DATASET_DIR / "train"
    train_imgs_dir, train_labels_dir = create_split_directory(train_dir)
    print(f"Train images directory: {train_imgs_dir}")
    print(f"Train labels directory: {train_labels_dir}")

    # Test games
    test_dir = DATASET_DIR / "test"
    test_imgs_dir, test_labels_dir = create_split_directory(test_dir)

    # Validation games
    val_dir = DATASET_DIR / "validation"
    val_imgs_dir, val_labels_dir = create_split_directory(val_dir)

    # Create labels mapping
    create_labels_mapping(DATASET_DIR)

    # Process train games
    for game_idx in tqdm(train_games, desc="Processing train games"):
        game_info = get_game_info(game_idx)
        process_game(
            game_info,
            train_imgs_dir,
            train_labels_dir,
            piece_paths=CHESS_PIECES_PATHS,
            board_colors=BOARD_COLORS_LIST,
        )

    # Process test games
    for game_idx in tqdm(test_games, desc="Processing test games"):
        game_info = get_game_info(game_idx)
        process_game(
            game_info,
            test_imgs_dir,
            test_labels_dir,
            piece_paths=CHESS_PIECES_PATHS,
            board_colors=BOARD_COLORS_LIST,
        )

    # Process validation games
    for game_idx in tqdm(validation_games, desc="Processing validation games"):
        game_info = get_game_info(game_idx)
        process_game(
            game_info,
            val_imgs_dir,
            val_labels_dir,
            piece_paths=CHESS_PIECES_PATHS,
            board_colors=BOARD_COLORS_LIST,
        )

    print("Processing completed!")


if __name__ == "__main__":

    # Get tournaments games
    selected_games = []
    for tournament_url in TOURNAMENT_URL_IDS:
        rounds_urls = get_tournament_rounds(tournament_url)
        groups_urls = get_tournament_round_groups(rounds_urls)
        selected_games.extend(get_games_in_rounds(groups_urls))

    # Filter for valid chess games (blitz and chess rules not chess360 or others)
    chess_games = [
        game
        for game in selected_games
        if game["time_class"] == "blitz" and game["rules"] == "chess"
    ]
    print(f"Number of valid chess games: {len(chess_games)}")

    # Sample chess games to reduce dataset size
    chess_games = random.sample(chess_games, N_CHESS_GAMES)
    print("Number of chess games after sampling: ", len(chess_games))

    # Holdout split
    games_idxs = list(range(len(chess_games)))
    train_games, test_games = train_test_split(
        games_idxs, test_size=TEST_TRAIN_RATIO, random_state=42
    )
    test_games, validation_games = train_test_split(
        test_games, test_size=VAL_TEST_RATIO, random_state=42
    )

    print(f"Number of train games: {len(train_games)}")
    print(f"Number of test games: {len(test_games)}")
    print(f"Number of validation games: {len(validation_games)}")

    process_games(
        train_games=train_games,
        test_games=test_games,
        validation_games=validation_games,
    )
